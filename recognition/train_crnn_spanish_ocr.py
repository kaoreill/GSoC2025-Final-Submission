# train_crnn_spanish_ocr.py (patched)
# ------------------------------------------------------------
# CRNN (CNN -> BiLSTM -> CTC) for Spanish OCR on Renaissance prints.
# Patches included:
# - Clean labels (NFC + optional ftfy, strip trailing " (4)", whitelist, collapse spaces)
# - Normalize images (mean=0.5, std=0.5)
# - Optional fixed width (--fixed_width) or keep-ratio + right-pad (default)
# - Early stopping (--early_stop_patience) & schedulers: OneCycle (default) or ReduceLROnPlateau
# - Save N bad preds each validation (--save_bad_preds_n)
# - dtype/contiguous safety before model forward (fixes conv2d/dtype issues)
# - Charset audit prints
# - Resume training from .pt/.pth (--resume_from), and skip pretrain (--finetune_only)
# - **FIX**: Correct per-sample CTC input lengths (no more using max T for all)
# - **AUG**: More realistic aug (illumination, morphology, pad jitter, softer perspective, gamma/brightness)
# - **CURRICULUM**: Pretrain aug strength decays midway to reduce domain gap
# - **VAL**: Finetune validates on REAL-only set
# - **SCHED**: Plateau scheduler steps on CER in both stages
# - **SAVE**: Un-normalize images before saving bad preds
# - **NORM**: Configurable norm layers (BN/GN/IN) via --norm
# - **DEVICE**: Auto-select CUDA/MPS/CPU (override with --device)
# - **DROPOUT**: Dropout between BiLSTMs for regularization
# ------------------------------------------------------------

import argparse
import os
import io
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import unicodedata
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, BatchSampler
# TensorBoard writer with safe fallback (handles setuptools/distutils issues)
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *a, **k): pass
        def add_scalar(self, *a, **k): pass
        def add_text(self, *a, **k): pass
        def add_hparams(self, *a, **k): pass
        def close(self, *a, **k): pass

import torchvision.transforms as T
from torch.nn.utils.clip_grad import clip_grad_norm_
from collections import OrderedDict

import re
try:
    from ftfy import fix_text as _fix_text
except Exception:
    _fix_text = None


# -----------------------------
# Utils & Repro
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Determinism off (faster on CPU)
    torch.use_deterministic_algorithms(False)
    os.environ["PYTHONHASHSEED"] = str(seed)


def nfc(s: str) -> str:
    # Normalize to NFC (keep diacritics, case)
    return unicodedata.normalize("NFC", s)


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# Label cleaning
# -----------------------------

_ALLOWED = set(
    " " +
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
    "abcdefghijklmnopqrstuvwxyz" +
    "áéíóúÁÉÍÓÚñÑ" +
    "0123456789" +
    "¡¿(),.-"
)
_allowed_class = "".join(sorted(_ALLOWED))
_allowed_escaped = "".join(re.escape(c) for c in _allowed_class)
_RE_NON_ALLOWED = re.compile(f"[^{_allowed_escaped}]")
_RE_TRAIL_PAREN_COUNTER = re.compile(r"\s*\(\s*\d+\s*\)\s*$")  # e.g., "hola (4)" -> "hola"

def clean_label(raw_stem: str) -> str:
    s = nfc(raw_stem)
    if _fix_text is not None:
        try:
            s = _fix_text(s)
        except Exception:
            pass
    s = _RE_TRAIL_PAREN_COUNTER.sub("", s)
    s = _RE_NON_ALLOWED.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\(\s*\)", "", s)
    return s


# -----------------------------
# Dataset scanning & charset
# -----------------------------

@dataclass
class Sample:
    path: Path
    label: str
    kind: str  # "real" | "synthetic" | "rare"


def scan_dir(root: Path, kind: str) -> List[Sample]:
    samples = []
    for p in root.rglob("*"):
        if p.is_file() and is_image(p):
            try:
                raw = p.stem
            except Exception:
                raw = str(p.stem)
            label = clean_label(raw)
            if len(label) == 0:
                continue
            samples.append(Sample(path=p, label=label, kind=kind))
    return samples


def split_80_10_10(samples: List[Sample], seed=42) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)
    n = len(idxs)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]
    return ([samples[i] for i in train_idx],
            [samples[i] for i in val_idx],
            [samples[i] for i in test_idx])


def extract_charset(samples: List[Sample]) -> List[str]:
    charset = set()
    for s in samples:
        for ch in s.label:
            charset.add(ch)
    if " " not in charset:
        charset.add(" ")
    return sorted(list(charset))


# -----------------------------
# Transforms (grayscale; aug policies)
# -----------------------------

class RandomGaussianNoise:
    def __init__(self, p=0.3, std_range=(2, 8)):
        self.p = p
        self.std_range = std_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img).astype(np.float32)
        std = random.uniform(*self.std_range)
        noise = np.random.normal(0, std, arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out)


class RandomJpegCompression:
    def __init__(self, p=0.25, quality_range=(30, 70)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("L")


class RandomRightPad:
    def __init__(self, max_pad=6, p=0.5):
        self.max_pad = max_pad
        self.p = p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p or self.max_pad <= 0:
            return img
        pad = random.randint(1, self.max_pad)
        # Manual right padding for PIL compatibility
        new_w = img.width + pad
        new_img = Image.new("L", (new_w, img.height), color=255)
        new_img.paste(img, (0, 0))
        return new_img


class RandomLeftPad:
    def __init__(self, max_pad=4, p=0.3):
        self.max_pad = max_pad
        self.p = p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p or self.max_pad <= 0:
            return img
        pad = random.randint(1, self.max_pad)
        # Manual left padding if ImageOps.expand does not support tuple border
        new_w = img.width + pad
        new_img = Image.new("L", (new_w, img.height), color=255)
        new_img.paste(img, (pad, 0))
        return new_img


class RandomIllumination:
    """Apply a low-frequency brightness gradient (vignetting/paper shading)."""
    def __init__(self, p=0.5, strength=0.15):
        self.p = p
        self.strength = strength
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img).astype(np.float32) / 255.0
        h, w = arr.shape[:2]
        # Horizontal or vertical gradient
        if random.random() < 0.5:
            a, b = 1.0 - self.strength*random.random(), 1.0 + self.strength*random.random()
            grad = np.linspace(a, b, w, dtype=np.float32)[None, :]
            grad = np.repeat(grad, h, axis=0)
        else:
            a, b = 1.0 - self.strength*random.random(), 1.0 + self.strength*random.random()
            grad = np.linspace(a, b, h, dtype=np.float32)[:, None]
            grad = np.repeat(grad, w, axis=1)
        out = np.clip(arr * grad, 0.0, 1.0)
        return Image.fromarray((out * 255).astype(np.uint8))


class RandomGamma:
    def __init__(self, p=0.3, gamma_range=(0.8, 1.2)):
        self.p = p
        self.gamma_range = gamma_range
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        g = random.uniform(*self.gamma_range)
        arr = np.array(img).astype(np.float32) / 255.0
        out = np.power(arr, g)
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


class RandomMorphology:
    """Slight erosion/dilation to mimic ink spread or worn type."""
    def __init__(self, p=0.25):
        self.p = p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        if random.random() < 0.5:
            # Erode (thinner)
            return img.filter(ImageFilter.MinFilter(size=3))
        else:
            # Dilate (thicker)
            return img.filter(ImageFilter.MaxFilter(size=3))


def _scale_tuple(t, s):
    return tuple(x*s for x in t)


def build_synthetic_augment(height: int, strength: float = 1.0):
    # Strength scales the magnitude of distortions (curriculum)
    deg = 2.0 * strength
    trans = (0.02*strength, 0.05*strength)
    scl = (1.0 - 0.03*strength, 1.0 + 0.03*strength)
    shear = 1.0 * strength
    persp = 0.08 * strength
    noise_hi = max(1.5, 6.0 * strength)
    return T.Compose([
        T.RandomAffine(degrees=deg, translate=trans, scale=scl, shear=shear, fill=255),
        T.RandomPerspective(distortion_scale=persp, p=0.25),
        RandomLeftPad(max_pad=4, p=0.3),
        RandomRightPad(max_pad=6, p=0.5),
        T.ColorJitter(brightness=0.2*strength, contrast=0.2*strength),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.25),
        RandomGaussianNoise(p=0.35, std_range=(1.5, noise_hi)),
        T.RandomAdjustSharpness(sharpness_factor=1.4, p=0.25),
        T.RandomAutocontrast(p=0.3),
        RandomIllumination(p=0.4, strength=0.15*strength),
        RandomGamma(p=0.25, gamma_range=(0.85, 1.15)),
        RandomMorphology(p=0.2),
        RandomJpegCompression(p=0.2, quality_range=(45, 80)),
    ])


def build_real_augment(height: int, strength: float = 1.0):
    deg = 0.8 * strength
    trans = (0.01*strength, 0.015*strength)
    scl = (1.0 - 0.015*strength, 1.0 + 0.015*strength)
    shear = 0.4 * strength
    noise_hi = max(0.8, 3.0 * strength)
    return T.Compose([
        T.RandomAffine(degrees=deg, translate=trans, scale=scl, shear=shear, fill=255),
        T.ColorJitter(brightness=0.12*strength, contrast=0.12*strength),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.12),
        RandomGaussianNoise(p=0.15, std_range=(0.8, noise_hi)),
        T.RandomAutocontrast(p=0.15),
        RandomIllumination(p=0.3, strength=0.12*strength),
        RandomGamma(p=0.2, gamma_range=(0.9, 1.1)),
    ])


# -----------------------------
# Image resize + right-pad
# -----------------------------

def resize_keep_ratio_right_pad(img: Image.Image, target_h: int, max_w: int) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(round(w * (target_h / h))))
    if new_w > max_w:
        new_w = max_w
    img = img.resize((new_w, target_h), Image.BILINEAR)
    canvas = Image.new("L", (new_w, target_h), color=255)
    canvas.paste(img, (0, 0))
    return canvas


# -----------------------------
# Dataset & Collate
# -----------------------------

class OCRWordDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        char_to_idx: Dict[str, int],
        target_h: int = 40,
        max_w_cap: int = 512,
        kind_aug: Optional[str] = None,  # "synthetic" | "real" | None
        fixed_width: int = 0,            # if >0, force (H, fixed_width)
        strength: float = 1.0,
    ):
        self.samples = samples
        self.char_to_idx = char_to_idx
        self.target_h = target_h
        self.max_w_cap = max_w_cap
        self.kind_aug = kind_aug
        self.fixed_width = fixed_width

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.kind_aug == "synthetic":
            self.aug = build_synthetic_augment(target_h, strength=strength)
        elif self.kind_aug == "real":
            self.aug = build_real_augment(target_h, strength=strength)
        else:
            self.aug = None

    def __len__(self):
        return len(self.samples)

    def encode_label(self, s: str) -> List[int]:
        return [self.char_to_idx[ch] for ch in s]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample.path).convert("L")
        if self.aug is not None:
            img = self.aug(img)

        if self.fixed_width and self.fixed_width > 0:
            img = img.resize((self.fixed_width, self.target_h), Image.BILINEAR)
        else:
            img = resize_keep_ratio_right_pad(img, self.target_h, self.max_w_cap)

        label = nfc(sample.label)
        label_ids = self.encode_label(label)

        x = self.to_tensor(img)
        x = self.normalize(x)
        w, _ = img.size
        return x, torch.tensor(label_ids, dtype=torch.long), label, w, sample.kind


def ocr_collate(batch):
    imgs, labels_ids, labels_str, widths, kinds = zip(*batch)
    max_w = max([img.shape[2] for img in imgs])
    padded = []
    for img in imgs:
        c, h, w = img.shape
        if w < max_w:
            pad = torch.ones((c, h, max_w - w), dtype=img.dtype)
            padded.append(torch.cat([img, pad], dim=2))
        else:
            padded.append(img)
    images = torch.stack(padded, dim=0)  # [B,1,H,W]
    target_lengths = torch.tensor([len(x) for x in labels_ids], dtype=torch.long)
    targets = torch.cat(labels_ids, dim=0) if len(labels_ids) > 0 else torch.empty(0, dtype=torch.long)
    # NEW: per-sample input lengths for CTC (width downsample factor = 4)
    # width downsample factor in CNN = 4 (two MaxPool2d(2,2))
    input_lengths = torch.tensor([max(1, w // 4) for w in widths], dtype=torch.long)
    return images, targets, target_lengths, labels_str, kinds, input_lengths


# -----------------------------
# Mixed batch sampler (aggressive rare oversampling)
# -----------------------------

class MixedKindBatchSampler(BatchSampler):
    def __init__(self, kinds: List[str], batch_size: int, drop_last: bool,
                 ratio_real: float = 0.0, ratio_synth: float = 0.8, ratio_rare: float = 0.2,
                 active_kinds: Optional[List[str]] = None, iters_per_epoch: Optional[int] = None, seed: int = 42):
        self.kinds = kinds
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.ratio_real = ratio_real
        self.ratio_synth = ratio_synth
        self.ratio_rare = ratio_rare
        self.active_kinds = active_kinds or ["synthetic", "rare"]
        self.seed = seed

        self.idx_by_kind = {k: [i for i, kk in enumerate(kinds) if kk == k] for k in ["real", "synthetic", "rare"]}
        self.rng = random.Random(seed)

        active_count = sum(len(self.idx_by_kind[k]) for k in self.active_kinds)
        if iters_per_epoch is None:
            iters_per_epoch = math.ceil(active_count / batch_size)
        self.iters_per_epoch = iters_per_epoch

    def __iter__(self):
        for _ in range(self.iters_per_epoch):
            n_rare = int(math.ceil(self.batch_size * self.ratio_rare)) if "rare" in self.active_kinds else 0
            n_real = int(math.ceil(self.batch_size * self.ratio_real)) if "real" in self.active_kinds else 0
            n_synth = self.batch_size - (n_rare + n_real)
            if "synthetic" not in self.active_kinds:
                n_synth = 0
                remaining = self.batch_size - (n_rare + n_real)
                for k in self.active_kinds:
                    if k == "rare":
                        n_rare += remaining; remaining = 0; break
                    if k == "real":
                        n_real += remaining; remaining = 0; break

            batch = []

            def sample_from(pool: List[int], n: int):
                if n <= 0: return []
                if len(pool) == 0: return []
                if len(pool) >= n:
                    return self.rng.sample(pool, n)
                return [self.rng.choice(pool) for _ in range(n)]

            batch += sample_from(self.idx_by_kind.get("rare", []), n_rare)
            batch += sample_from(self.idx_by_kind.get("real", []), n_real)
            batch += sample_from(self.idx_by_kind.get("synthetic", []), n_synth)

            if len(batch) > self.batch_size:
                batch = batch[:self.batch_size]
            elif len(batch) < self.batch_size:
                all_active = []
                for k in self.active_kinds:
                    all_active += self.idx_by_kind.get(k, [])
                if len(all_active) > 0:
                    fill = [self.rng.choice(all_active) for _ in range(self.batch_size - len(batch))]
                    batch += fill
            yield batch

    def __len__(self):
        return self.iters_per_epoch


# -----------------------------
# CRNN Model
# -----------------------------

def Norm2dFactory(kind: str, num_channels: int):
    if kind == "bn":
        return nn.BatchNorm2d(num_channels)
    elif kind == "gn":
        # cap groups to <= channels and power of two-ish
        groups = min(32, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups //= 2
        return nn.GroupNorm(groups, num_channels)
    elif kind == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    else:
        raise ValueError(f"Unknown norm kind: {kind}")


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out)
        return out


class CRNN(nn.Module):
    def __init__(self, num_classes: int, img_h: int = 40, norm: str = "bn"):
        super().__init__()
        N = lambda c: Norm2dFactory(norm, c)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), N(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H: 40->20

            nn.Conv2d(64, 128, 3, 1, 1), N(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H: 20->10

            nn.Conv2d(128, 256, 3, 1, 1), N(256), nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, 1), N(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H: 10->5

            nn.Conv2d(256, 512, 3, 1, 1), N(512), nn.ReLU(True),

            nn.Conv2d(512, 512, 3, 1, 1), N(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H: 5->2 (for H=40) / 6->3 (for H=48)

            nn.Conv2d(512, 512, (2, 1), (2, 1), (0, 0)),  # H: 2->1 (or 3->1 due to stride)
            nn.ReLU(True),
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            nn.Dropout(0.3),
            BidirectionalLSTM(256, 256, num_classes),
        )

    def forward(self, x):
        features = self.cnn(x)                 # [B, 512, 1, W']
        features = features.squeeze(2).permute(0, 2, 1)  # [B, W', 512]
        y = self.rnn(features)                 # [B, W', C]
        y = y.permute(1, 0, 2)                 # [T, B, C]
        return y


# -----------------------------
# CTC helpers: greedy decode
# -----------------------------

def greedy_decode(logits: torch.Tensor, blank_idx: int, idx_to_char: Dict[int, str]) -> List[str]:
    with torch.no_grad():
        pred = logits.argmax(-1)  # [T, B]
        Tt, B = pred.shape
        results = []
        for b in range(B):
            seq = pred[:, b].tolist()
            prev = None
            chars = []
            for t in seq:
                if t == blank_idx:
                    prev = None
                    continue
                if t == prev:
                    continue
                chars.append(idx_to_char[t])
                prev = t
            results.append("".join(chars))
        return results


# -----------------------------
# Metrics: CER/WER (Unicode-sensitive)
# -----------------------------

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            temp = dp[j]
            cb = b[j - 1]
            if ca == cb:
                dp[j] = prev
            else:
                dp[j] = min(prev + 1, dp[j] + 1, dp[j - 1] + 1)
            prev = temp
    return dp[lb]


def cer(preds: List[str], gts: List[str]) -> float:
    num, den = 0, 0
    for p, g in zip(preds, gts):
        p = nfc(p); g = nfc(g)
        num += levenshtein(p, g)
        den += len(g)
    return (num / max(1, den)) if den > 0 else 0.0


def wer(preds: List[str], gts: List[str]) -> float:
    num, den = 0, 0
    for p, g in zip(preds, gts):
        ptoks = p.split(); gtoks = g.split()
        num += levenshtein(" ".join(ptoks), " ".join(gtoks))
        den += max(1, len(gtoks))
    return num / max(1, den)


# -----------------------------
# Train / Eval
# -----------------------------

def run_epoch(model: nn.Module,
              loader: DataLoader,
              optimizer,
              scheduler,
              device,
              criterion,
              idx_to_char,
              blank_idx,
              phase: str,
              writer: Optional[SummaryWriter],
              step_base: int,
              epoch: int,
              args,
              out_dir: Path):
    model.train(phase == "train")
    total_loss = 0.0
    total_cer_n, total_cer_d = 0, 0
    total_wer_n, total_wer_d = 0, 0
    step = step_base

    save_bad_dir = None
    saved_bad = 0
    if phase == "val" and args.save_bad_preds_n > 0:
        save_bad_dir = out_dir / "bad_preds"
        save_bad_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        images, targets, target_lengths, labels_str, kinds, input_lengths = batch

        # dtype/contiguity guard
        images = images.to(device, dtype=torch.float32).contiguous()
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        input_lengths = input_lengths.to(device)

        if phase == "train":
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)  # [T, B, C]
        Tt, B, C = logits.shape
        log_probs = logits.log_softmax(2)

        # --- PATCH: cap per-sample input lengths to the actual T ---
        input_lengths = torch.minimum(
            input_lengths,
            torch.full((B,), Tt, dtype=torch.long, device=input_lengths.device)
        )

        # Use true per-sample lengths (no more tying to max T)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        if phase == "train":
            # backprop first, then clip + step + (maybe) scheduler
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if scheduler is not None and isinstance(scheduler, OneCycleLR):
                scheduler.step()

        total_loss += loss.item()

        with torch.no_grad():
            pred_str = greedy_decode(logits, blank_idx, idx_to_char)
            # accumulate CER/WER
            total_cer_n += sum(levenshtein(nfc(p), nfc(g)) for p, g in zip(pred_str, labels_str))
            total_cer_d += sum(len(nfc(g)) for g in labels_str)
            total_wer_n += sum(levenshtein(" ".join(nfc(p).split()), " ".join(nfc(g).split()))
                               for p, g in zip(pred_str, labels_str))
            total_wer_d += sum(max(1, len(nfc(g).split())) for g in labels_str)

            # save a few bad preds during validation
            if save_bad_dir is not None and saved_bad < args.save_bad_preds_n:
                for i, (gt, pr) in enumerate(zip(labels_str, pred_str)):
                    if saved_bad >= args.save_bad_preds_n:
                        break
                    if gt != pr:
                        im = images[i].detach().cpu()
                        # UN-NORMALIZE before saving
                        im = im * 0.5 + 0.5
                        im = (im.clamp(0, 1) * 255).byte().squeeze(0).numpy()
                        Image.fromarray(im, mode="L").save(
                            save_bad_dir / f"ep{epoch:03d}_i{step:06d}_GT_{gt}_PRED_{pr}.png"
                        )
                        saved_bad += 1

        if writer is not None and (step % 50 == 0):
            writer.add_scalar(f"{phase}/loss", loss.item(), step)

        if writer is not None and (step % 500 == 0):
            for i in range(min(8, len(pred_str))):
                writer.add_text(f"{phase}/sample_{i}",
                                f"kind={kinds[i]} | gt={labels_str[i]} | pred={pred_str[i]}", step)

        step += 1

    avg_loss = total_loss / max(1, len(loader))
    avg_cer = (total_cer_n / max(1, total_cer_d)) if total_cer_d > 0 else 0.0
    avg_wer = (total_wer_n / max(1, total_wer_d)) if total_wer_d > 0 else 0.0
    return avg_loss, avg_cer, avg_wer, step



# -----------------------------
# Data assembly helpers
# -----------------------------

def assemble_splits(real_dirs: List[Path], synth_dirs: List[Path], rare_dirs: List[Path], seed=42):
    all_real, all_synth, all_rare = [], [], []
    for d in real_dirs:  all_real  += scan_dir(d, "real")
    for d in synth_dirs: all_synth += scan_dir(d, "synthetic")
    for d in rare_dirs:  all_rare  += scan_dir(d, "rare")

    real_tr, real_val, real_te = split_80_10_10(all_real, seed)
    synth_tr, synth_val, synth_te = split_80_10_10(all_synth, seed)
    rare_tr, rare_val, rare_te = split_80_10_10(all_rare, seed)

    train_all = real_tr + synth_tr + rare_tr
    val_all = real_val + synth_val + rare_val
    test_all = real_te + synth_te + rare_te

    pretrain_train = synth_tr + rare_tr
    pretrain_val = synth_val + rare_val  # validate pretrain on synthetic-like by default
    finetune_train = real_tr
    finetune_val = real_val              # **REAL-only validation for finetune**

    return {
        "all": {"train": train_all, "val": val_all, "test": test_all},
        "pretrain": {"train": pretrain_train, "val": pretrain_val},
        "finetune": {"train": finetune_train, "val": finetune_val},
        "groups": {"real": (real_tr, real_val, real_te),
                   "synthetic": (synth_tr, synth_val, synth_te),
                   "rare": (rare_tr, rare_val, rare_te)}
    }


def build_char_maps(samples: List[Sample], out_dir: Path):
    charset = extract_charset(samples)
    char_to_idx = {ch: i for i, ch in enumerate(charset)}
    blank_idx = len(charset)
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    (out_dir / "charset.txt").write_text("".join(charset), encoding="utf-8")
    with open(out_dir / "charset.json", "w", encoding="utf-8") as f:
        json.dump({"char_to_idx": char_to_idx, "blank_idx": blank_idx}, f, ensure_ascii=False, indent=2)
    return charset, char_to_idx, idx_to_char, blank_idx


def find_unknown(chars, charset):
    return sorted(set(chars) - set(charset))


def collect_text(samples):
    return "".join(s.label for s in samples)


def remap_output_layer_from_resume(model, resume_state, old_charset, new_charset, old_blank, new_blank):
    """
    Copy rows of the final linear layer from the old checkpoint into the new model
    according to character identity. New characters are randomly initialized.
    Also moves the old blank row to the new blank index.
    """
    # Names of the final layer in your CRNN:
    W_key = "rnn.1.linear.weight"
    b_key = "rnn.1.linear.bias"

    if W_key not in resume_state or b_key not in resume_state:
        print("[charset] Could not find final layer weights in checkpoint; skipping remap.")
        return

    old_W = resume_state[W_key]          # [old_C, 512]
    old_b = resume_state[b_key]          # [old_C]
    new_W = model.rnn[1].linear.weight   # [new_C, 512]
    new_b = model.rnn[1].linear.bias     # [new_C]

    with torch.no_grad():
        new_W.copy_(new_W)
        new_b.copy_(new_b)

        old_idx = {ch: i for i, ch in enumerate(old_charset)}
        new_idx = {ch: i for i, ch in enumerate(new_charset)}

        common = sorted(set(old_charset) & set(new_charset))
        copied = 0
        for ch in common:
            i_old = old_idx[ch]
            i_new = new_idx[ch]
            new_W[i_new].copy_(old_W[i_old])
            new_b[i_new].copy_(old_b[i_old])
            copied += 1

        print(f"[charset] Copied {copied}/{len(old_charset)} rows from old head.")

        if old_blank is not None and new_blank is not None:
            if old_blank < old_W.shape[0] and new_blank < new_W.shape[0]:
                new_W[new_blank].copy_(old_W[old_blank])
                new_b[new_blank].copy_(old_b[old_blank])
                print(f"[charset] Mapped BLANK from old idx {old_blank} -> new idx {new_blank}.")


def remap_output_layer_from_tensors(model, old_W, old_b,
                                    old_charset, new_charset,
                                    old_blank, new_blank):
    new_lin = model.rnn[1].linear
    with torch.no_grad():
        old_idx = {ch: i for i, ch in enumerate(old_charset)}
        new_idx = {ch: i for i, ch in enumerate(new_charset)}
        common = set(old_charset) & set(new_charset)
        copied = 0
        for ch in common:
            i_old = old_idx[ch]; i_new = new_idx[ch]
            new_lin.weight[i_new].copy_(old_W[i_old])
            new_lin.bias[i_new].copy_(old_b[i_old])
            copied += 1
        print(f"[charset] Copied {copied}/{len(old_charset)} rows from old head.")
        if old_blank is not None and new_blank is not None:
            if old_blank < old_W.shape[0] and new_blank < new_lin.weight.shape[0]:
                new_lin.weight[new_blank].copy_(old_W[old_blank])
                new_lin.bias[new_blank].copy_(old_b[old_blank])
                print(f"[charset] Mapped BLANK {old_blank} -> {new_blank}.")

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="CRNN Spanish OCR (CTC) - Renaissance Prints [patched]")
    parser.add_argument("--real_dirs", nargs="*", type=str, default=[], help="Directories with REAL images (filenames are labels).")
    parser.add_argument("--synthetic_dirs", nargs="*", type=str, default=[], help="Directories with SYNTHETIC images.")
    parser.add_argument("--rare_dirs", nargs="*", type=str, default=[], help="Directories with RARE-CHAR images (oversampled).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for checkpoints, logs, charset, etc.")
    parser.add_argument("--height", type=int, default=40, help="Target image height (e.g., 40 or 48).")
    parser.add_argument("--max_width_cap", type=int, default=512, help="Max width cap at resize (safety).")
    parser.add_argument("--fixed_width", type=int, default=0, help="If >0, resize to (H, fixed_width) instead of keep-ratio.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain_epochs", type=int, default=40)
    parser.add_argument("--finetune_epochs", type=int, default=25)
    parser.add_argument("--max_lr", type=float, default=3e-3, help="OneCycleLR max LR.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0, help="0 for Windows CPU-friendly.")
    parser.add_argument("--export_onnx", type=str, default="", help="Path to export ONNX after training.")
    parser.add_argument("--early_stop_patience", type=int, default=0, help="0 disables early stopping.")
    parser.add_argument("--scheduler", type=str, default="onecycle", choices=["onecycle", "plateau"], help="LR scheduler type.")
    parser.add_argument("--save_bad_preds_n", type=int, default=0, help="Save up to N wrong predictions per validation epoch.")
    # Resume flags
    parser.add_argument("--resume_from", type=str, default="", help="Path to .pt/.pth checkpoint with {'model': state_dict, 'charset': [...], 'blank_idx': int}.")
    parser.add_argument("--finetune_only", action="store_true", help="Skip pretrain; only run finetune stage (useful when resuming).")
    parser.add_argument("--charset_mode", type=str, default="resume",
                    choices=["resume", "data", "union", "file"],
                    help="How to build charset when resuming: use checkpoint ('resume'), "
                         "infer from data ('data'), union of both ('union'), or load from file ('file').")
    parser.add_argument("--charset_file", type=str, default="",
                        help="Path to a text file with one-line charset (used when --charset_mode file).")
    # New toggles
    parser.add_argument("--norm", type=str, default="bn", choices=["bn","gn","in"], help="Normalization layers in CNN.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"], help="Device override.")

    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "has_mps") and torch.has_mps and torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    real_dirs = [Path(p) for p in args.real_dirs]
    synth_dirs = [Path(p) for p in args.synthetic_dirs]
    rare_dirs = [Path(p) for p in args.rare_dirs]

    print("Scanning datasets ...")
    splits = assemble_splits(real_dirs, synth_dirs, rare_dirs, seed=args.seed)

    # Build charset from ALL samples (train+val+test)
    all_samples = splits["all"]["train"] + splits["all"]["val"] + splits["all"]["test"]

    # ----- RESUME HANDLING: if provided, prefer checkpoint's charset & blank -----
    resume_state = None
    resume_charset = None
    resume_blank = None
    if args.resume_from and os.path.isfile(args.resume_from):
        print(f"[resume] Loading checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu")
        # Accept various key conventions
        resume_state = ckpt.get("model_state_dict", ckpt.get("model", None))
        if resume_state is None and isinstance(ckpt, dict):
            possible_keys = [k for k, v in ckpt.items() if torch.is_tensor(v) or isinstance(v, np.ndarray)]
            print(f"[resume] Could not find 'model' or 'model_state_dict' keys; keys in ckpt: {list(ckpt.keys())[:10]}")
        # DataParallel prefix cleanup
        if resume_state is not None and any(k.startswith("module.") for k in resume_state.keys()):
            resume_state = OrderedDict((k.replace("module.", "", 1), v) for k, v in resume_state.items())
        if "charset" in ckpt:
            resume_charset = list(ckpt["charset"])
            print(f"[resume] Loaded charset from checkpoint (len={len(resume_charset)})")
        if "blank_idx" in ckpt:
            resume_blank = int(ckpt["blank_idx"])
            print(f"[resume] Loaded blank_idx from checkpoint: {resume_blank}")

    # ----- Build data-inferred charset for reference -----
    data_charset = sorted(set(extract_charset(all_samples)))
    if " " not in data_charset:
        data_charset = [" "] + [c for c in data_charset if c != " "]

    # ----- Decide FINAL charset per mode -----
    if args.charset_mode == "resume":
        if resume_charset is None:
            print("[charset] No charset in checkpoint; falling back to DATA.")
            final_charset = data_charset
        else:
            final_charset = resume_charset

    elif args.charset_mode == "data":
        final_charset = data_charset

    elif args.charset_mode == "union":
        base = set(resume_charset) if resume_charset is not None else set()
        final_charset = sorted(base | set(data_charset))

    elif args.charset_mode == "file":
        assert args.charset_file, "--charset_file is required when --charset_mode file"
        text = Path(args.charset_file).read_text(encoding="utf-8")
        final_charset = list(text.rstrip("\n"))

    else:
        raise ValueError(f"Unknown charset_mode: {args.charset_mode}")

    # Ensure space is present (nice to have)
    if " " not in final_charset:
        final_charset = [" "] + [c for c in final_charset if c != " "]

    # Build maps
    char_to_idx = {ch: i for i, ch in enumerate(final_charset)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    blank_idx = len(final_charset)

    # Persist to out_dir
    (out_dir / "charset.txt").write_text("".join(final_charset), encoding="utf-8")
    with open(out_dir / "charset.json", "w", encoding="utf-8") as f:
        json.dump({"char_to_idx": char_to_idx, "blank_idx": blank_idx}, f, ensure_ascii=False, indent=2)

    print(f"Using FINAL charset size: {len(final_charset)} (blank idx={blank_idx})")
    print("Unknown in train (w.r.t final):", find_unknown(collect_text(splits['all']['train']), final_charset))
    print("Unknown in val   (w.r.t final):", find_unknown(collect_text(splits['all']['val']),   final_charset))
    print("Unknown in test  (w.r.t final):", find_unknown(collect_text(splits['all']['test']),  final_charset))

    # Helper to build datasets
    def make_ds(sample_list: List[Sample], aug_mode: Optional[str], strength: float = 1.0):
        return OCRWordDataset(
            samples=sample_list,
            char_to_idx=char_to_idx,
            target_h=args.height,
            max_w_cap=args.max_width_cap,
            kind_aug=aug_mode,
            fixed_width=args.fixed_width,
            strength=strength,
        )

    # Pretrain datasets (synthetic + rare) with strong aug
    pretrain_train_samples = splits["pretrain"]["train"]

    class MixedAugDataset(Dataset):
        def __init__(self, samples, char_to_idx, h, mw, fixed_width, strength=1.0):
            self.samples = samples
            self.syn_ds = OCRWordDataset(samples=[], char_to_idx=char_to_idx, target_h=h, max_w_cap=mw, kind_aug="synthetic", fixed_width=fixed_width, strength=strength)
            self.real_ds = OCRWordDataset(samples=[], char_to_idx=char_to_idx, target_h=h, max_w_cap=mw, kind_aug="real", fixed_width=fixed_width, strength=strength)
            self.none_ds = OCRWordDataset(samples=[], char_to_idx=char_to_idx, target_h=h, max_w_cap=mw, kind_aug=None, fixed_width=fixed_width, strength=strength)
            self.char_to_idx = char_to_idx
            self.h = h; self.mw = mw; self.fixed_width = fixed_width
            self.strength = strength
        def set_strength(self, strength: float):
            self.strength = strength
            self.syn_ds.aug = build_synthetic_augment(self.h, strength)
            self.real_ds.aug = build_real_augment(self.h, strength)
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            s = self.samples[i]
            ds = self.syn_ds if (s.kind in ("synthetic", "rare")) else (self.real_ds if s.kind == "real" else self.none_ds)
            img = Image.open(s.path).convert("L")
            if ds.aug is not None: img = ds.aug(img)
            if self.fixed_width and self.fixed_width > 0:
                img = img.resize((self.fixed_width, ds.target_h), Image.BILINEAR)
            else:
                img = resize_keep_ratio_right_pad(img, ds.target_h, ds.max_w_cap)
            label = nfc(s.label)
            label_ids = [char_to_idx[ch] for ch in label]
            x = ds.to_tensor(img)
            x = ds.normalize(x)
            return x, torch.tensor(label_ids, dtype=torch.long), label, img.size[0], s.kind

    pretrain_kinds = [s.kind for s in pretrain_train_samples]
    pretrain_ds = MixedAugDataset(pretrain_train_samples, char_to_idx, args.height, args.max_width_cap, args.fixed_width, strength=1.0)

    # Finetune: real only, light aug
    finetune_train_real = [s for s in splits["finetune"]["train"] if s.kind == "real"]
    finetune_ds = make_ds(finetune_train_real, aug_mode="real", strength=1.0)

    # Validation
    val_pretrain_ds = make_ds(splits["pretrain"]["val"], aug_mode=None)   # pretrain val: synth+rare
    val_finetune_ds = make_ds(splits["finetune"]["val"], aug_mode=None)   # finetune val: real-only

    # Test: no aug
    test_ds = make_ds(splits["all"]["test"], aug_mode=None)

    # Loaders + Samplers
    pretrain_batch_sampler = MixedKindBatchSampler(
        kinds=pretrain_kinds,
        batch_size=args.batch_size,
        drop_last=False,
        ratio_real=0.0, ratio_synth=0.9, ratio_rare=0.1,
        active_kinds=["synthetic", "rare"],
        seed=args.seed
    )
    pretrain_loader = DataLoader(pretrain_ds, batch_sampler=pretrain_batch_sampler,
                                 num_workers=args.num_workers, collate_fn=ocr_collate)

    finetune_loader = DataLoader(finetune_ds, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, collate_fn=ocr_collate)

    val_pretrain_loader = DataLoader(val_pretrain_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=ocr_collate)
    val_finetune_loader = DataLoader(val_finetune_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=ocr_collate)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=ocr_collate)

    num_classes = len(final_charset) + 1  # +blank

    model = CRNN(num_classes=num_classes, img_h=args.height, norm=args.norm).to(device)

    # Keys of the final linear layer
    W_key = "rnn.1.linear.weight"
    b_key = "rnn.1.linear.bias"

    old_W = None
    old_b = None

    if resume_state is not None:
        # Keep a copy of old head tensors for remap
        if W_key in resume_state and b_key in resume_state:
            old_W = resume_state[W_key].clone()
            old_b = resume_state[b_key].clone()

            # If output size changed, remove the conflicting keys so load_state_dict won't error
            out_rows_ckpt = old_W.shape[0]
            out_rows_new = len(final_charset) + 1  # +blank
            if out_rows_ckpt != out_rows_new:
                resume_state = resume_state.copy()
                resume_state.pop(W_key, None)
                resume_state.pop(b_key, None)

        print("[resume] Loading model weights (non-strict) …")
        missing, unexpected = model.load_state_dict(resume_state, strict=False)
        print(f"[resume] missing keys: {len(missing)} | unexpected: {len(unexpected)}")

        # If we have the old head and charsets differ (or size changed), remap rows
        if (old_W is not None) and (resume_charset is not None) and (resume_charset != final_charset):
            old_blank = resume_blank if resume_blank is not None else len(resume_charset)
            new_blank = blank_idx
            print("[charset] Final charset differs from checkpoint; remapping final layer …")
            remap_output_layer_from_tensors(model, old_W, old_b,
                                            resume_charset, final_charset,
                                            old_blank, new_blank)

    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    writer = SummaryWriter(log_dir=str(out_dir / "logs"))

    best_val_cer = float("inf")
    global_step = 0

    # ----------------- PRETRAIN -----------------
    if not args.finetune_only:
        print("\n=== PRETRAIN (synthetic + rare) ===")
        opt = optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
        if args.scheduler == "onecycle":
            steps_per_epoch = len(pretrain_loader)
            sched = OneCycleLR(opt, max_lr=args.max_lr, epochs=args.pretrain_epochs,
                               steps_per_epoch=max(1, steps_per_epoch), pct_start=0.1, anneal_strategy='cos')
        else:
            # Step on validation CER later (quality metric)
            sched = ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5)

        no_improve = 0  # early stop counter

        for epoch in range(1, args.pretrain_epochs + 1):
            # Curriculum: soften aug halfway
            strength = 0.7 if epoch > (args.pretrain_epochs * 0.5) else 1.0
            pretrain_ds.set_strength(strength)

            t0 = time.time()
            tr_loss, tr_cer, tr_wer, global_step = run_epoch(
                model, pretrain_loader, opt, sched if args.scheduler == "onecycle" else None, device, criterion,
                idx_to_char={i: ch for i, ch in enumerate(final_charset)}, blank_idx=blank_idx,
                phase="train", writer=writer, step_base=global_step, epoch=epoch, args=args, out_dir=out_dir
            )
            with torch.no_grad():
                val_loss, val_cer, val_wer, _ = run_epoch(
                    model, val_pretrain_loader, opt, None, device, criterion,
                    idx_to_char={i: ch for i, ch in enumerate(final_charset)}, blank_idx=blank_idx,
                    phase="val", writer=writer, step_base=global_step, epoch=epoch, args=args, out_dir=out_dir
                )

            if args.scheduler == "plateau":
                sched.step(val_cer)  # step on CER (not loss)

            dt = time.time() - t0
            print(f"[Pretrain][Epoch {epoch}/{args.pretrain_epochs}] "
                  f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_CER={val_cer:.4f} val_WER={val_wer:.4f} ({dt:.1f}s)")

            writer.add_scalar("val/CER", val_cer, global_step)
            writer.add_scalar("val/WER", val_wer, global_step)

            improved = val_cer < (best_val_cer - 1e-4)
            if improved:
                best_val_cer = val_cer
                torch.save({"model": model.state_dict(),
                            "charset": final_charset,
                            "blank_idx": blank_idx},
                           out_dir / "checkpoints" / "best_pretrain.pt")
                no_improve = 0
            else:
                no_improve += 1

            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                print(f"Early stopping triggered (pretrain) after {epoch} epochs.")
                break
    else:
        print("\n=== Skipping PRETRAIN (finetune_only) ===")

    # ----------------- FINETUNE -----------------
    print("\n=== FINETUNE (real only) ===")
    opt = optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    if args.scheduler == "onecycle":
        steps_per_epoch = len(finetune_loader)
        sched = OneCycleLR(opt, max_lr=args.max_lr, epochs=args.finetune_epochs,
                           steps_per_epoch=max(1, steps_per_epoch), pct_start=0.1, anneal_strategy='cos')
    else:
        sched = ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5)

    no_improve = 0

    for epoch in range(1, args.finetune_epochs + 1):
        t0 = time.time()
        tr_loss, tr_cer, tr_wer, global_step = run_epoch(
            model, finetune_loader, opt, sched if args.scheduler == "onecycle" else None, device, criterion,
            idx_to_char={i: ch for i, ch in enumerate(final_charset)}, blank_idx=blank_idx,
            phase="train", writer=writer, step_base=global_step, epoch=epoch, args=args, out_dir=out_dir
        )
        with torch.no_grad():
            val_loss, val_cer, val_wer, _ = run_epoch(
                model, val_finetune_loader, opt, None, device, criterion,
                idx_to_char={i: ch for i, ch in enumerate(final_charset)}, blank_idx=blank_idx,
                phase="val", writer=writer, step_base=global_step, epoch=epoch, args=args, out_dir=out_dir
            )

        if args.scheduler == "plateau":
            sched.step(val_cer)  # step on CER (target metric)

        dt = time.time() - t0
        print(f"[Finetune][Epoch {epoch}/{args.finetune_epochs}] "
              f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
              f"val_CER={val_cer:.4f} val_WER={val_wer:.4f} ({dt:.1f}s)")

        writer.add_scalar("val/CER", val_cer, global_step)
        writer.add_scalar("val/WER", val_wer, global_step)

        improved = val_cer < (best_val_cer - 1e-4)
        if improved:
            best_val_cer = val_cer
            torch.save({"model": model.state_dict(),
                        "charset": final_charset,
                        "blank_idx": blank_idx},
                       out_dir / "checkpoints" / "best_finetune.pt")
            no_improve = 0
        else:
            no_improve += 1

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"Early stopping triggered (finetune) after {epoch} epochs.")
            break

    # ----------------- TEST -----------------
    print("\n=== TEST ===")
    with torch.no_grad():
        test_loss, test_cer, test_wer, _ = run_epoch(
            model, test_loader, opt, None, device, criterion,
            idx_to_char={i: ch for i, ch in enumerate(final_charset)}, blank_idx=blank_idx,
            phase="test", writer=writer, step_base=global_step, epoch=0, args=args, out_dir=out_dir
        )
    print(f"[TEST] loss={test_loss:.4f} CER={test_cer:.4f} WER={test_wer:.4f}")
    writer.add_hparams(
        {
            "height": args.height,
            "fixed_width": args.fixed_width,
            "batch_size": args.batch_size,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "max_lr": args.max_lr,
            "scheduler": args.scheduler,
            "norm": args.norm,
        },
        {
            "test/loss": test_loss,
            "test/CER": test_cer,
            "test/WER": test_wer,
        }
    )

    # Save final model
    torch.save({"model": model.state_dict(), "charset": final_charset, "blank_idx": blank_idx},
               out_dir / "checkpoints" / "final.pt")

    # Optional: export ONNX
    if args.export_onnx:
        print("Exporting ONNX ...")
        model.eval()
        dummy_w = args.fixed_width if (args.fixed_width and args.fixed_width > 0) else 256
        dummy = torch.rand(1, 1, args.height, dummy_w, dtype=torch.float32)
        torch.onnx.export(
            model, dummy, args.export_onnx,
            input_names=["images"], output_names=["logits_TxBxC"],
            dynamic_axes={"images": {0: "batch", 3: "width"},
                          "logits_TxBxC": {0: "time", 1: "batch"}},
            opset_version=13
        )
        print(f"ONNX saved to: {args.export_onnx}")

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
