import json
import os
import sys
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from pathlib import Path
from functools import lru_cache
import re
import time
import hashlib
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageTk
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
import onnxruntime as ort



# === Configuration ===
# Adjust these paths to your environment
POPPLER_PATH  = r'C:/Users/katej/OneDrive/Documents/Downloads/Release-24.08.0-0/poppler-24.08.0/Library/bin/'
CRAFT_SCRIPT  = 'RenAIssance_CRNN_OCR_Kate_OReilly_orig/CRAFT_Model/CRAFT/BoundBoxFunc/test.py'
CRAFT_WEIGHTS = 'RenAIssance_CRNN_OCR_Kate_OReilly_orig/CRAFT_Model/CRAFT/BoundBoxFunc/weights/craft_mlt_25k.pth'

# Project directories (absolute, space-free recommended)
BASE           = Path('RenAIssance_CRNN_OCR_Kate_OReilly_orig').resolve()
PROCESSED_DIR  = (BASE / 'data' / 'processed_pages').resolve()     # images fed into CRAFT
RESULT_DIR     = (BASE / 'data' / 'bounding_boxes').resolve()      # CRAFT outputs (res_page_*.txt, overlays)
DATASET_ROOT   = (BASE / 'data' / 'data_6').resolve()              # dataset output root

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_ROOT.mkdir(parents=True, exist_ok=True)


# === Utilities ===

def load_page_bytes(pdf_bytes: bytes, page: int, dpi: int) -> np.ndarray:
    """Load a single PDF page (1-based index) to BGR image."""
    pil_img = convert_from_bytes(
        pdf_bytes, dpi=dpi, first_page=page, last_page=page, poppler_path=POPPLER_PATH
    )[0]
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def split_image_vertically(img: np.ndarray):
    """Split a two-page spread into left/right halves."""
    h, w = img.shape[:2]
    mid = w // 2
    return img[:, :mid], img[:, mid:]


def _ensure_3c(img: np.ndarray) -> np.ndarray:
    """Ensure 3-channel BGR image."""
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def sort_boxes(boxes):
    """Sort quads by (top y, then left x)."""
    def key(b):
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        return (min(ys), min(xs))
    return sorted(boxes, key=key)


def parse_boxes(txt_path: Path):
    """
    Parse CRAFT result txt file. Expect 8 numbers per line: x1 y1 x2 y2 x3 y3 x4 y4
    Returns list of 4-point polygons [(x,y)*4].
    """
    boxes = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            vals = [int(float(v)) for v in line.replace(',', ' ').split()]
            if len(vals) == 8:
                boxes.append([(vals[i], vals[i+1]) for i in range(0, 8, 2)])
    return boxes


def group_boxes_by_lines(boxes, y_thresh=35):
    """
    Group word boxes into lines: boxes with similar top y belong to same line.
    Inside each line, boxes are left→right.
    """
    groups = []
    for b in sort_boxes(boxes):
        y_top = min(p[1] for p in b)
        placed = False
        for line in groups:
            ref_y = min(p[1] for p in line[0])
            if abs(y_top - ref_y) <= y_thresh:
                line.append(b)
                placed = True
                break
        if not placed:
            groups.append([b])
    for line in groups:
        line.sort(key=lambda bb: min(p[0] for p in bb))
    return groups


def find_result_triplets(result_dir: Path):
    """
    Return list of (overlay_img_path, txt_path, page_idx) ordered by page index.
    overlay_img_path prefers 'res_page_XXX.(jpg|png)' from RESULT_DIR, else falls back
    to the preprocessed 'page_XXX.png' from PROCESSED_DIR so labeler always has an image.
    """
    txts = sorted(result_dir.glob('res_page_*.txt'))
    triplets = []
    for t in txts:
        stem = t.stem  # res_page_001
        img_jpg = result_dir / f'{stem}.jpg'
        img_png = result_dir / f'{stem}.png'
        if img_jpg.exists():
            img = img_jpg
        elif img_png.exists():
            img = img_png
        else:
            # Fallback to the preprocessed PNG generated for CRAFT input
            num = stem.split('_')[-1]
            fallback = PROCESSED_DIR / f'page_{num}.png'
            img = fallback if fallback.exists() else None
        if img and img.exists():
            idx = int(stem.split('_')[-1])
            triplets.append((img, t, idx))
    triplets.sort(key=lambda x: x[2])
    return triplets


def tokens_from_line(text: str):
    """
    Split on whitespace, keep punctuation attached, preserve case.
    Empty/whitespace-only -> returns [].
    """
    if text is None:
        return []
    stripped = text.strip()
    if not stripped:
        return []
    return re.split(r'\s+', stripped)


def safe_class_dir(s: str) -> str:
    """
    Make a folder name safe on Windows while preserving most punctuation and case.
    Replace forbidden chars with _UXXXX (unicode code point).
    Avoid trailing dots/spaces and reserved device names.
    """
    forbidden = '<>:"/\\|?*'
    out = []
    for ch in s:
        if ch in forbidden or ord(ch) < 32:
            out.append(f'_U{ord(ch):04X}')
        else:
            out.append(ch)
    name = ''.join(out).strip()
    if name in ('', '.', '..'):
        # Use a content-derived hash to keep classes unique but readable
        name = f'_EMPTY_{hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]}'
    name = name.rstrip(' .')
    reserved = {'CON','PRN','AUX','NUL','COM1','COM2','COM3','COM4','COM5','COM6','COM7','COM8','COM9',
                'LPT1','LPT2','LPT3','LPT4','LPT5','LPT6','LPT7','LPT8','LPT9'}
    if name.upper() in reserved:
        name = f'_{name}_'
    return name


def rect_from_quad(quad, W, H):
    """
    Axis-aligned rectangle bounds from quad, clamped to image.
    Returns (x0, y0, x1, y1) with x1/y1 exclusive for numpy slicing; or None if invalid.
    """
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x0 = max(0, min(xs))
    y0 = max(0, min(ys))
    x1 = min(W, max(xs) + 1)
    y1 = min(H, max(ys) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)

# ==== ONNX OCR utilities ====

def ocr_resize_keep_ratio_right_pad(pil_img: Image.Image, target_h: int, max_w: int = 512) -> Image.Image:
    """Resize to target_h, keep aspect, right-pad to max batch width later."""
    w, h = pil_img.size
    new_w = max(1, int(round(w * (target_h / h))))
    new_w = min(new_w, max_w)
    return pil_img.resize((new_w, target_h), Image.BILINEAR)

def ocr_preprocess_batch(pil_list: list, height: int, fixed_width: int = 0, max_w_cap: int = 512) -> np.ndarray:
    """To NCHW float32 normalized [B,1,H,W], padded on the right."""
    proc = []
    widths = []
    for im in pil_list:
        if fixed_width and fixed_width > 0:
            im = im.resize((fixed_width, height), Image.BILINEAR)
        else:
            im = ocr_resize_keep_ratio_right_pad(im, height, max_w_cap)
        arr = np.array(im.convert("L")).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        proc.append(arr)
        widths.append(im.size[0])

    H = height
    Wmax = max(widths) if widths else (fixed_width if fixed_width else 256)
    B = len(proc)
    batch = np.ones((B, 1, H, Wmax), dtype=np.float32)
    for i, arr in enumerate(proc):
        w = arr.shape[1]
        batch[i, 0, :, :w] = arr
    return batch

def ocr_greedy_decode(logits_TxBxC: np.ndarray, blank_idx: int, idx_to_char: dict) -> list:
    """CTC greedy decode; logits_TxBxC from ONNX output."""
    pred = logits_TxBxC.argmax(-1)  # [T,B]
    T, B = pred.shape
    out = []
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
        out.append(unicodedata.normalize("NFC", "".join(chars)))
    return out

def load_charset_from_dir(charset_dir: Path):
    with open(charset_dir / "charset.txt", "r", encoding="utf-8") as f:
        charset = list(f.read())
    with open(charset_dir / "charset.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    blank_idx = meta["blank_idx"]
    idx_to_char = {i: ch for i, ch in enumerate(charset)}
    return idx_to_char, blank_idx

class CRNNOCR:
    """Lightweight ONNX runner for your CRNN-CTC model."""
    def __init__(self, onnx_path: Path, charset_dir: Path, height: int, fixed_width: int = 0):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.idx_to_char, self.blank_idx = load_charset_from_dir(charset_dir)
        self.height = height
        self.fixed_width = fixed_width

    def infer_tokens(self, crops: list) -> list:
        """
        crops: list of PIL images (grayscale or RGB)
        returns: list[str] predictions
        """
        batch = ocr_preprocess_batch(crops, self.height, self.fixed_width)
        (logits,) = self.sess.run(["logits_TxBxC"], {"images": batch})
        return ocr_greedy_decode(logits, self.blank_idx, self.idx_to_char)

# === Page Break Wizard ===
class PageBreakWizard(tk.Toplevel):
    """Let the user mark the last line of each page (maps transcript lines to pages)."""
    def __init__(self, parent, transcript_lines, page_count, on_done):
        super().__init__(parent)
        self.title("Assign Transcript Lines to Pages")
        self.geometry("780x600")
        self.transcript_lines = transcript_lines
        self.page_count = page_count
        self.on_done = on_done

        self.break_indices = []  # line index of last line per page except the last page

        # UI listbox with all transcript lines
        self.listbox = tk.Listbox(self, font=("Consolas", 12), selectmode=tk.BROWSE)
        for i, line in enumerate(transcript_lines):
            self.listbox.insert(tk.END, f"{i+1:4} | {line}")
        self.listbox.pack(fill='both', expand=True, padx=12, pady=(12, 6))

        nav = ttk.Frame(self)
        nav.pack(fill='x', padx=12, pady=(0, 12))
        self.add_break_btn = ttk.Button(nav, text="Mark end of page", command=self.set_break)
        self.add_break_btn.pack(side='left')
        ttk.Button(nav, text="Undo last break", command=self.undo_break).pack(side='left', padx=8)
        ttk.Button(nav, text="Finish", command=self.finish).pack(side='right')
        self.page_status = ttk.Label(nav, text=f"Setting page 1 / {self.page_count}")
        self.page_status.pack(side='right', padx=(0, 10))

        self.listbox.bind("<<ListboxSelect>>", lambda e: self._update_status())
        self._update_status()

    def _update_status(self):
        # Color: past pages blue, current selection yellow
        total = self.listbox.size()
        for j in range(total):
            self.listbox.itemconfig(j, background='white')
        start = 0
        for idx in self.break_indices:
            for j in range(start, idx + 1):
                self.listbox.itemconfig(j, background='#d9f5ff')
            start = idx + 1
        sel = self.listbox.curselection()
        cur_start = self.break_indices[-1] + 1 if self.break_indices else 0
        cur_end = sel[0] if sel else cur_start - 1
        for j in range(cur_start, min(cur_end + 1, total)):
            self.listbox.itemconfig(j, background='#fff7b3')
        page_num = len(self.break_indices) + 1
        self.page_status.config(text=f"Setting page {page_num} / {self.page_count}")

    def set_break(self):
        if len(self.break_indices) >= self.page_count - 1:
            messagebox.showinfo("Done", "All page breaks are already set.")
            return
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Select a line", "Select a line to mark as page end.")
            return
        idx = sel[0]
        if self.break_indices and idx <= self.break_indices[-1]:
            messagebox.showwarning("Order error", "This break must be after the previous break.")
            return
        self.break_indices.append(idx)
        self._update_status()
        if len(self.break_indices) == self.page_count - 1:
            self.finish()

    def undo_break(self):
        if self.break_indices:
            self.break_indices.pop()
            self._update_status()

    def finish(self):
        # Build (start, end) for each page (inclusive)
        ranges = []
        prev = 0
        for idx in self.break_indices:
            ranges.append((prev, idx))
            prev = idx + 1
        ranges.append((prev, len(self.transcript_lines) - 1))
        if len(ranges) != self.page_count:
            messagebox.showerror("Mismatch", "Number of page splits does not match page count.")
            return
        self.on_done(ranges)
        self.destroy()


# === Per-Page Line Labeler ===
class LineLabeler(tk.Toplevel):
    """
    Per-page labeler with alignment:
      - Left: overlay image with word boxes (color coded)
      - Right: per-line transcription entries + status (tokens vs boxes)
      - Controls: Skip line, Prev/Next line, Save draft, Approve page
    """
    def __init__(self, parent, page_idx: int, img_path: Path, line_groups, lines, on_approve, on_savedraft):
        super().__init__(parent)
        self.title(f"Label page {page_idx:03d}: {img_path.name}")
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        w = min(screen_w - 60, 1400)
        h = min(screen_h - 60, 900)
        self.geometry(f"{w}x{h}")

        self.page_idx = page_idx
        self.img_bgr = cv2.imread(str(img_path))
        self.line_groups = line_groups  # list[list[quad]]
        self.on_approve = on_approve
        self.on_savedraft = on_savedraft

        # model
        self.cur = 0
        self.lines = list(lines) + [""] * max(0, (len(self.line_groups) - len(lines)))
        self.skipped = [False] * len(self.line_groups)

        # Layout
        container = ttk.Frame(self)
        container.pack(fill='both', expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(0, weight=1)

        # Canvas with overlay
        self.canvas = tk.Canvas(container, bg='#0f1f3d', highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=(12, 6), pady=12, sticky='nsew')
        self.canvas.bind('<Configure>', self.redraw)

        # Right panel
        right = ttk.Frame(container)
        right.grid(row=0, column=1, padx=(6, 12), pady=12, sticky='nsew')
        right.rowconfigure(1, weight=1)

        ttk.Label(right, text="Transcript per line (whitespace → words):", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky='w', pady=(0, 6))

        self.entries_frame = ttk.Frame(right)
        self.entries_frame.grid(row=1, column=0, sticky='nsew')

        self.entries_canvas = tk.Canvas(self.entries_frame, bg='#faf6ef', highlightthickness=0)
        self.entries_scroll = ttk.Scrollbar(self.entries_frame, orient="vertical", command=self.entries_canvas.yview)
        self.entries_inner = ttk.Frame(self.entries_canvas)
        self.entries_inner.bind("<Configure>", lambda e: self.entries_canvas.configure(scrollregion=self.entries_canvas.bbox("all")))
        self.entries_canvas.create_window((0, 0), window=self.entries_inner, anchor="nw")
        self.entries_canvas.configure(yscrollcommand=self.entries_scroll.set)
        self.entries_canvas.pack(side="left", fill="both", expand=True)
        self.entries_scroll.pack(side="right", fill="y")

        self.entry_widgets = []
        self.status_widgets = []
        for i in range(len(self.line_groups)):
            row = ttk.Frame(self.entries_inner)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=f"{i+1:02}", width=3).pack(side='left')
            e = ttk.Entry(row)
            e.pack(side='left', fill='x', expand=True, padx=4)
            if i < len(self.lines) and self.lines[i]:
                e.insert(0, self.lines[i])
            e.bind("<KeyRelease>", lambda _ev, idx=i: self._on_entry_changed(idx))
            self.entry_widgets.append(e)

            status = ttk.Label(row, text="", width=18)
            status.pack(side='left', padx=(6,0))
            self.status_widgets.append(status)

        # Footer / navigation
        nav = ttk.Frame(self)
        nav.pack(fill='x', padx=12, pady=(0, 12))
        ttk.Button(nav, text="Prev line", command=self.prev_line).pack(side='left')
        ttk.Button(nav, text="Next line", command=self.next_line).pack(side='left', padx=6)
        ttk.Button(nav, text="Skip line", command=self.skip_line).pack(side='left', padx=6)
        ttk.Button(nav, text="Save draft", command=self.save_draft).pack(side='left', padx=6)
        ttk.Button(nav, text="Approve page ✅", command=self.approve_page).pack(side='left', padx=6)
        self.status = ttk.Label(nav, text=f"Line 1 of {len(self.line_groups)}")
        self.status.pack(side='right')

        # Initialize status/overlay
        for i in range(len(self.line_groups)):
            self._refresh_line_status(i)
        self.focus_set()

    # --- Drawing/logic helpers ---

    def _scaled_preview(self):
        img = self.img_bgr
        if img is None:
            return None, 1.0
        ch, cw = img.shape[:2]
        cW, cH = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(cW / cw, cH / ch)
        view = cv2.resize(img, (int(cw * scale), int(ch * scale)))
        return view, scale

    def redraw(self, _evt=None):
        view, scale = self._scaled_preview()
        if view is None:
            return
        for i, group in enumerate(self.line_groups):
            # color logic: red = mismatch, cyan = current, green = current-ok, gray = skipped, light gray = neutral
            if self.skipped[i]:
                color = (180, 180, 180)
            else:
                tokens = tokens_from_line(self.entry_widgets[i].get())
                mismatch = len(tokens) != len(group) and len(tokens) > 0
                color = (0, 230, 0) if (i == self.cur and not mismatch) else ((0, 200, 200) if i == self.cur else ((0, 0, 255) if mismatch else (210, 210, 210)))
            thick = 3 if i == self.cur else 1
            for box in group:
                pts = np.array([(int(x * scale), int(y * scale)) for (x, y) in box], np.int32).reshape((-1, 1, 2))
                cv2.polylines(view, [pts], isClosed=True, color=color, thickness=thick)
        imgRGB = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(imgRGB))
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, image=self.tkimg, anchor='center')
        self.status.config(text=f"Line {self.cur+1} of {len(self.line_groups)}")

    def _on_entry_changed(self, idx):
        self._refresh_line_status(idx)
        if idx == self.cur:
            self.redraw()

    def _refresh_line_status(self, idx):
        tokens = tokens_from_line(self.entry_widgets[idx].get())
        box_count = len(self.line_groups[idx])
        lbl = self.status_widgets[idx]
        if self.skipped[idx]:
            lbl.config(text="SKIPPED", foreground="#666")
        else:
            if len(tokens) == 0:
                lbl.config(text=f"0 tok / {box_count} box", foreground="#CC6600")  # empty -> treated as skipped at export
            elif len(tokens) == box_count:
                lbl.config(text=f"{len(tokens)} = {box_count}", foreground="#0A8F08")
            else:
                lbl.config(text=f"{len(tokens)} ≠ {box_count}", foreground="#CC0000")

    # --- Navigation & actions ---

    def prev_line(self):
        self.cur = max(0, self.cur - 1)
        self.entry_widgets[self.cur].focus_set()
        self.redraw()

    def next_line(self):
        self.cur = min(len(self.line_groups) - 1, self.cur + 1)
        self.entry_widgets[self.cur].focus_set()
        self.redraw()

    def skip_line(self):
        self.skipped[self.cur] = True
        self.entry_widgets[self.cur].delete(0, tk.END)
        self.entry_widgets[self.cur].state(['disabled'])
        self._refresh_line_status(self.cur)
        self.next_line()

    def _collect_page_data(self):
        """
        Returns list of per-line dicts:
        { 'skipped': bool, 'tokens': [..], 'boxes': [quad,...] }
        Empty line text is treated as skipped at export time.
        """
        data = []
        for i, group in enumerate(self.line_groups):
            toks = [] if self.skipped[i] else tokens_from_line(self.entry_widgets[i].get())
            data.append({'skipped': self.skipped[i] or (len(toks) == 0),
                         'tokens': toks,
                         'boxes': group})
        return data

    def save_draft(self):
        if self.on_savedraft:
            self.on_savedraft(self.page_idx, self._collect_page_data())
        messagebox.showinfo("Saved", f"Draft saved for page {self.page_idx:03d}.")

    def approve_page(self):
        # Validate: all non-skipped lines must have equal token/box counts
        data = self._collect_page_data()
        bad = []
        for i, item in enumerate(data):
            if not item['skipped'] and len(item['tokens']) != len(item['boxes']):
                bad.append(i+1)
        if bad:
            messagebox.showerror("Cannot approve", f"Token/box mismatch on lines: {', '.join(map(str, bad))}.\n"
                                                   f"Fix or skip those lines to approve.")
            return
        if self.on_approve:
            self.on_approve(self.page_idx, data)   # parent sets approved=True
        self.destroy()


# === Main App ===
class PDFApp(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=0)
        self.parent = parent
        self.parent.title("PDF OCR Preprocessor → Word Dataset Exporter")
        self.parent.geometry("1320x880")
        self.parent.minsize(1120, 780)

        # State
        self.pdf_bytes = None
        self.page_count = 1
        self.split_page_count = 1
        self.split_spreads = tk.BooleanVar(value=False)

        # Preprocessing configuration toggles
        self.cfg = {'h': 10, 'k': 55, 'clip': 3.0, 'area': 50, 'line_y_thresh': 35}
        self.var_dn = tk.BooleanVar(value=True)
        self.var_bg = tk.BooleanVar(value=True)
        self.var_ct = tk.BooleanVar(value=True)
        self.var_bn = tk.BooleanVar(value=True)
        self.var_cl = tk.BooleanVar(value=True)
        self.var_fb = tk.BooleanVar(value=True)

        # Labeling/export state
        self._after_id = None
        self.current = None
        self.transcript_page_ranges = []  # [(start,end), ...]
        self.all_transcript_lines = []
        self._labeling_triplets = []
        self._labeling_index = 0
        # page_sessions[page_idx] = { 'approved': bool, 'lines': [ {skipped, tokens, boxes}, ... ] }
        self.page_sessions = {}

        self._setup_style()
        self._build_ui()

    # --- Styling ---
    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        cream = '#faf6ef'
        navy  = '#12264a'
        blue  = '#0077ff'
        mid   = '#20305a'
        line  = '#285080'

        style.configure('.', font=('Segoe UI', 11), background=cream, foreground=navy)
        self.parent.configure(background=cream)

        style.configure('Nav.TFrame', background=cream)
        style.configure('Main.TFrame', background=cream)

        style.configure('Sidebar.TLabelframe', background=navy, foreground=cream, bordercolor=line, borderwidth=2)
        style.configure('Sidebar.TLabelframe.Label', background=navy, foreground='#00b0f0', font=('Segoe UI', 12, 'bold'))
        style.configure('Sidebar.TCheckbutton', background=navy, foreground=cream)
        style.configure('Sidebar.TLabel', background=navy, foreground=cream)
        style.configure('Accent.TButton', background=blue, foreground=cream, borderwidth=0, padding=10, font=('Segoe UI', 11, 'bold'))
        style.map('Accent.TButton', background=[('active', mid), ('pressed', line)], foreground=[('active', cream), ('pressed', cream)])

        style.configure('TLabelframe', background=cream, foreground=line, bordercolor=line, borderwidth=2)
        style.configure('TLabelframe.Label', background=cream, foreground=blue, font=('Segoe UI', 12, 'bold'))

    # --- UI ---
    def _build_ui(self):
    # Top bar (file + view controls)
        nav = ttk.Frame(self, style='Nav.TFrame', padding=(18, 12, 18, 10))
        nav.pack(side='top', fill='x')

        ttk.Label(nav, text="📄 PDF:").pack(side='left', padx=(0, 6))
        self.pdf_entry = ttk.Entry(nav, width=42)
        self.pdf_entry.pack(side='left', padx=(0, 8))
        ttk.Button(nav, text="Browse…", style='Accent.TButton', command=self.load_pdf).pack(side='left', padx=(0, 10))

        ttk.Label(nav, text="Page:").pack(side='left')
        self.page_spin = ttk.Spinbox(nav, from_=1, to=1, width=6, command=self._on_spin)
        self.page_spin.pack(side='left', padx=(4, 10))

        ttk.Label(nav, text="DPI:").pack(side='left')
        self.dpi_scale = ttk.Scale(nav, from_=50, to=600, orient='horizontal', length=320)
        self.dpi_scale.set(300)
        self.dpi_scale.pack(side='left', padx=(6, 6))
        self.dpi_scale.bind('<ButtonRelease-1>', lambda e: self._on_scale())

        # Main split
        main = ttk.Frame(self, style='Main.TFrame')
        main.pack(side='top', fill='both', expand=True)

        # ==== Scrollable Sidebar (options + actions) ====
        # Container for canvas + scrollbar
        sidebar_container = ttk.Labelframe(
            main, text="Options & Actions",
            style='Sidebar.TLabelframe', padding=0
        )
        sidebar_container.pack(side='left', fill='y', padx=(18, 10), pady=(0, 12))

        # Canvas + vertical scrollbar
        sidebar_canvas = tk.Canvas(
            sidebar_container, highlightthickness=0, bd=0,
            bg="#1e3a8a"
        )
        sidebar_scroll = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)

        sidebar_canvas.pack(side='left', fill='y', expand=False)
        sidebar_scroll.pack(side='right', fill='y')

        # Actual sidebar frame placed inside the canvas
        sidebar = ttk.Frame(sidebar_canvas, style='Sidebar.TFrame', padding=(16, 14, 16, 18))
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor='nw')

        # Keep canvas scrollregion sized to inner content
        def _resize_scrollregion(_evt=None):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))
            # Lock the sidebar width to the canvas width so widgets wrap properly
            sidebar_canvas.itemconfigure(sidebar_window, width=sidebar_canvas.winfo_width())

        sidebar.bind("<Configure>", _resize_scrollregion)
        sidebar_container.bind("<Configure>", lambda e: _resize_scrollregion())

        # Mousewheel support (Windows/Mac/Linux)
        def _on_mousewheel(event):
            # Windows / MacOS
            if event.delta:
                sidebar_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else:
                # Linux (event.num 4=up, 5=down)
                if event.num == 4:
                    sidebar_canvas.yview_scroll(-3, "units")
                elif event.num == 5:
                    sidebar_canvas.yview_scroll(3, "units")

        sidebar_canvas.bind_all("<MouseWheel>", _on_mousewheel)      # Windows/Mac
        sidebar_canvas.bind_all("<Button-4>", _on_mousewheel)        # Linux up
        sidebar_canvas.bind_all("<Button-5>", _on_mousewheel)        # Linux down

        # ---- Sidebar content (unchanged; now add to `sidebar`) ----
        ttk.Checkbutton(sidebar, text="Denoise", variable=self.var_dn, style='Sidebar.TCheckbutton').pack(anchor='w', pady=2)
        ttk.Label(sidebar, text="Strength (h):", style='Sidebar.TLabel').pack(anchor='w')
        h_scale = ttk.Scale(sidebar, from_=1, to=30, orient='horizontal', length=160,
                            command=lambda e: self._debounced_update('h', int(float(e))))
        h_scale.set(self.cfg['h']); h_scale.pack(anchor='w', pady=(0, 8))

        ttk.Checkbutton(sidebar, text="Background remove", variable=self.var_bg, style='Sidebar.TCheckbutton').pack(anchor='w', pady=2)
        ttk.Label(sidebar, text="Median kernel (odd):", style='Sidebar.TLabel').pack(anchor='w')
        k_scale = ttk.Scale(sidebar, from_=3, to=101, orient='horizontal', length=160,
                            command=lambda e: self._debounced_update('k', max(3, int(float(e)) | 1)))
        k_scale.set(self.cfg['k']); k_scale.pack(anchor='w', pady=(0, 8))

        ttk.Checkbutton(sidebar, text="Contrast (CLAHE)", variable=self.var_ct, style='Sidebar.TCheckbutton').pack(anchor='w', pady=2)
        ttk.Label(sidebar, text="Clip limit:", style='Sidebar.TLabel').pack(anchor='w')
        clip_scale = ttk.Scale(sidebar, from_=1, to=10, orient='horizontal', length=160,
                            command=lambda e: self._debounced_update('clip', float(e)))
        clip_scale.set(self.cfg['clip']); clip_scale.pack(anchor='w', pady=(0, 8))

        ttk.Checkbutton(sidebar, text="Binarize (Otsu)", variable=self.var_bn, style='Sidebar.TCheckbutton').pack(anchor='w', pady=2)
        ttk.Checkbutton(sidebar, text="Close (morph)", variable=self.var_cl, style='Sidebar.TCheckbutton').pack(anchor='w', pady=2)
        ttk.Checkbutton(sidebar, text="Filter blobs (min area)", variable=self.var_fb, style='Sidebar.TCheckbutton').pack(anchor='w', pady=2)
        ttk.Label(sidebar, text="Min area:", style='Sidebar.TLabel').pack(anchor='w')
        area_scale = ttk.Scale(sidebar, from_=1, to=2000, orient='horizontal', length=160,
                            command=lambda e: self._debounced_update('area', int(float(e))))
        area_scale.set(self.cfg['area']); area_scale.pack(anchor='w', pady=(0, 8))

        ttk.Label(sidebar, text="Line grouping Y-thresh:", style='Sidebar.TLabel').pack(anchor='w', pady=(8, 0))
        line_scale = ttk.Scale(sidebar, from_=10, to=80, orient='horizontal', length=160,
                            command=lambda e: self._debounced_update('line_y_thresh', int(float(e))))
        line_scale.set(self.cfg['line_y_thresh']); line_scale.pack(anchor='w', pady=(0, 12))

        ttk.Checkbutton(sidebar, text="Split two-page spreads", variable=self.split_spreads, style='Sidebar.TCheckbutton').pack(anchor='w', pady=(4, 10))
        self.split_spreads.trace_add('write', lambda *_: self._on_split_toggle())

        ttk.Button(sidebar, text="Process Preview", style='Accent.TButton', command=self.process_preview).pack(fill='x', pady=(6, 6))
        self.craft_btn = ttk.Button(sidebar, text="Run CRAFT", style='Accent.TButton', command=self.run_craft)
        self.craft_btn.pack(fill='x')

        ttk.Separator(sidebar).pack(fill='x', pady=10)

        # Export controls
        ttk.Label(sidebar, text="Export:", style='Sidebar.TLabel').pack(anchor='w', pady=(6, 4))
        self.export_btn = ttk.Button(sidebar, text="Export Dataset (all pages approved)", style='Accent.TButton', command=self.export_dataset)
        self.export_btn.pack(fill='x')
        self.export_btn.state(['disabled'])  # enabled only when every page is approved

        ttk.Label(sidebar, text="Progress:", style='Sidebar.TLabel').pack(anchor='w', pady=(16, 4))
        self.progress = ttk.Progressbar(sidebar, orient='horizontal', length=160, mode='determinate')
        self.progress.pack(anchor='w', fill='x')
        self.craft_status = ttk.Label(sidebar, text="", style='Sidebar.TLabel')
        self.craft_status.pack(anchor='w', pady=(4, 0))

        # --- OCR (ONNX) mode controls ---
        ttk.Separator(sidebar).pack(fill='x', pady=10)
        ttk.Label(sidebar, text="OCR (no transcript):", style='Sidebar.TLabel').pack(anchor='w', pady=(6, 4))

        self.var_use_onnx = tk.BooleanVar(value=False)
        ttk.Checkbutton(sidebar, text="Use ONNX model instead of transcript", variable=self.var_use_onnx, style='Sidebar.TCheckbutton').pack(anchor='w', pady=2)

        ocr_box = ttk.Frame(sidebar)
        ocr_box.pack(fill='x', pady=(6, 4))

        ttk.Label(ocr_box, text="ONNX path:", style='Sidebar.TLabel').grid(row=0, column=0, sticky='w')
        self.ocr_onnx_entry = ttk.Entry(ocr_box, width=30)
        self.ocr_onnx_entry.grid(row=0, column=1, sticky='ew', padx=(6, 0))
        ttk.Button(ocr_box, text="Browse", command=self._browse_onnx).grid(row=0, column=2, padx=(6,0))
        ocr_box.columnconfigure(1, weight=1)

        ttk.Label(ocr_box, text="Charset dir:", style='Sidebar.TLabel').grid(row=1, column=0, sticky='w', pady=(6,0))
        self.ocr_charset_entry = ttk.Entry(ocr_box, width=30)
        self.ocr_charset_entry.grid(row=1, column=1, sticky='ew', padx=(6, 0), pady=(6,0))
        ttk.Button(ocr_box, text="Browse", command=self._browse_charset).grid(row=1, column=2, padx=(6,0), pady=(6,0))

        ttk.Label(ocr_box, text="Model height:", style='Sidebar.TLabel').grid(row=2, column=0, sticky='w', pady=(6,0))
        self.ocr_height_spin = ttk.Spinbox(ocr_box, from_=16, to=128, width=6)
        self.ocr_height_spin.delete(0, tk.END); self.ocr_height_spin.insert(0, '32')
        self.ocr_height_spin.grid(row=2, column=1, sticky='w', padx=(6,0), pady=(6,0))

        ttk.Label(ocr_box, text="Fixed width (0=flex):", style='Sidebar.TLabel').grid(row=3, column=0, sticky='w', pady=(6,0))
        self.ocr_fixed_spin = ttk.Spinbox(ocr_box, from_=0, to=1024, width=6)
        self.ocr_fixed_spin.delete(0, tk.END); self.ocr_fixed_spin.insert(0, '0')
        self.ocr_fixed_spin.grid(row=3, column=1, sticky='w', padx=(6,0), pady=(6,0))

        # ==== Previews (left: preprocessed, right: CRAFT overlay sample) ====
        previews = ttk.Frame(main, style='Main.TFrame')
        previews.pack(side='left', fill='both', expand=True, padx=(0, 18), pady=(0, 12))
        previews.rowconfigure(0, weight=1)
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)

        left_lab = ttk.Labelframe(previews, text="Preview (preprocessed)", padding=8)
        left_lab.grid(row=0, column=0, sticky='nsew', padx=(0, 10), pady=4)
        self.left_lbl = ttk.Label(left_lab, anchor='center', background='#faf6ef')
        self.left_lbl.pack(expand=True, fill='both', padx=4, pady=4)

        right_lab = ttk.Labelframe(previews, text="CRAFT Output (first page)", padding=8)
        right_lab.grid(row=0, column=1, sticky='nsew', padx=(0, 0), pady=4)
        self.right_lbl = ttk.Label(right_lab, anchor='center', background='#faf6ef')
        self.right_lbl.pack(expand=True, fill='both', padx=4, pady=4)

        # Status bar
        self.status = ttk.Label(self.parent, text="Ready", anchor='w', relief='sunken', font=('Segoe UI', 10))
        self.status.pack(side='bottom', fill='x')

    def _browse_onnx(self):
        path = filedialog.askopenfilename(filetypes=[("ONNX Model", "*.onnx")])
        if path:
            self.ocr_onnx_entry.delete(0, tk.END)
            self.ocr_onnx_entry.insert(0, path)

    def _browse_charset(self):
        path = filedialog.askdirectory()
        if path:
            self.ocr_charset_entry.delete(0, tk.END)
            self.ocr_charset_entry.insert(0, path)

    def _load_ocr_model(self) -> CRNNOCR:
        onnx_path = self.ocr_onnx_entry.get().strip()
        charset_dir = self.ocr_charset_entry.get().strip()
        if not onnx_path or not os.path.isfile(onnx_path):
            raise FileNotFoundError("Please provide a valid ONNX model path.")
        if not charset_dir or not os.path.isdir(charset_dir):
            raise FileNotFoundError("Please provide a valid charset directory (contains charset.txt & charset.json).")
        try:
            h = int(self.ocr_height_spin.get())
        except Exception:
            h = 32
        try:
            fw = int(self.ocr_fixed_spin.get())
        except Exception:
            fw = 0
        return CRNNOCR(Path(onnx_path), Path(charset_dir), height=h, fixed_width=fw)

    def run_ocr_and_label(self):
        """
        1) Find CRAFT results.
        2) For each page, crop words from PROCESSED_DIR page image.
        3) Batch through ONNX and decode.
        4) Pre-fill LineLabeler entries with predicted tokens (review/edit as needed).
        """
        triplets = find_result_triplets(RESULT_DIR)
        if not triplets:
            messagebox.showinfo("No results", "Unable to find CRAFT result images/txt.")
            return

        try:
            ocr = self._load_ocr_model()
        except Exception as e:
            messagebox.showerror("OCR setup error", str(e))
            return

        # Reset sessions
        self.page_sessions = {}
        self._labeling_triplets = triplets
        self._labeling_index = 0

        # Optionally, we could auto-approve; instead we open reviewers with pre-filled tokens
        self._open_next_ocr_labeler(ocr)

    def _open_next_ocr_labeler(self, ocr: CRNNOCR):
        if self._labeling_index >= len(self._labeling_triplets):
            self._update_export_button_state()
            messagebox.showinfo("OCR pass complete", "All pages visited. Approve pages or export.")
            return

        img_path, txt_path, page_idx = self._labeling_triplets[self._labeling_index]
        boxes = parse_boxes(txt_path)
        line_groups = group_boxes_by_lines(boxes, y_thresh=self.cfg['line_y_thresh'])

        # Crop words from the preprocessed page image (same as export source)
        src_img = PROCESSED_DIR / f'page_{page_idx:03d}.png'
        img_cv = cv2.imread(str(src_img))
        if img_cv is None:
            # fall back to overlay image
            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                messagebox.showwarning("Missing page image", f"Cannot read {src_img} or {img_path}. Skipping page {page_idx}.")
                self._labeling_index += 1
                self._open_next_ocr_labeler(ocr)
                return
        H, W = img_cv.shape[:2]

        # Build crops per line, run OCR per line (preserve grouping)
        prefilled_lines = []
        for group in line_groups:
            crops = []
            rects = []
            for quad in group:
                r = rect_from_quad(quad, W, H)
                if r is None:
                    continue
                x0, y0, x1, y1 = r
                crop = img_cv[y0:y1, x0:x1]
                if crop is None or crop.size == 0:
                    continue
                crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                rects.append(quad)

            if crops:
                preds = ocr.infer_tokens(crops)
                prefilled_lines.append({'skipped': False, 'tokens': preds, 'boxes': rects})
            else:
                prefilled_lines.append({'skipped': True, 'tokens': [], 'boxes': []})

        # Open the labeler with predictions pre-filled
        def on_savedraft(cb_page_idx, page_data):
            self.page_sessions[cb_page_idx] = {
                'approved': self.page_sessions.get(cb_page_idx, {}).get('approved', False),
                'lines': page_data
            }
            self._update_export_button_state()

        def on_approve(cb_page_idx, page_data):
            self.page_sessions[cb_page_idx] = {'approved': True, 'lines': page_data}
            self._labeling_index += 1
            self._update_export_button_state()
            self._open_next_ocr_labeler(ocr)

        # Convert predictions to “lines” text for the right-hand entries
        line_texts = [" ".join(item['tokens']) if (not item['skipped'] and item['tokens']) else "" for item in prefilled_lines]

        # Keep the same overlay & groups
        LineLabeler(self.parent, page_idx, img_path, line_groups, line_texts, on_approve, on_savedraft)

    # --- Preview pipeline ---
    def preprocess_img(self, img_bgr: np.ndarray) -> np.ndarray:
        """Apply preprocessing used for CRAFT input and preview (returns 3-channel BGR)."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if self.var_dn.get():
            gray = cv2.fastNlMeansDenoising(gray, h=self.cfg['h'])
        if self.var_bg.get():
            gray = cv2.divide(gray, cv2.medianBlur(gray, self.cfg['k']), scale=255)
        if self.var_ct.get():
            clahe = cv2.createCLAHE(clipLimit=self.cfg['clip'], tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] if self.var_bn.get() else gray
        if self.var_cl.get():
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        if self.var_fb.get():
            _, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
            mask = np.zeros_like(bw)
            for i in range(1, len(stats)):
                if stats[i, cv2.CC_STAT_AREA] >= self.cfg['area']:
                    mask[labels == i] = 255
            bw = mask
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    def _on_spin(self): 
        self.refresh_preview()

    def _on_scale(self): 
        self.refresh_preview()

    def _debounced_update(self, key, val):
        self.cfg[key] = val
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(200, self.refresh_preview)

    def _on_split_toggle(self):
        if not self.pdf_bytes:
            return
        self.split_page_count = self.page_count * 2 if self.split_spreads.get() else self.page_count
        self.page_spin.configure(to=self.split_page_count)
        self.page_spin.delete(0, tk.END)
        self.page_spin.insert(0, '1')
        self.refresh_preview()

    # --- File / preview ---

    def load_pdf(self):
        """Open a PDF file and initialize page counts."""
        path = filedialog.askopenfilename(filetypes=[('PDF', '*.pdf')])
        if not path:
            return
        self.pdf_entry.delete(0, tk.END)
        self.pdf_entry.insert(0, path)
        with open(path, 'rb') as f:
            self.pdf_bytes = f.read()
        info = pdfinfo_from_bytes(self.pdf_bytes, poppler_path=POPPLER_PATH)
        self.page_count = int(info.get('Pages', 1))
        self.split_page_count = self.page_count * 2 if self.split_spreads.get() else self.page_count
        self.page_spin.configure(to=self.split_page_count)
        self.page_spin.delete(0, tk.END)
        self.page_spin.insert(0, '1')
        self.status.config(text=f'Loaded "{Path(path).name}" ({self.page_count} pages)')
        self.refresh_preview()

    @lru_cache(maxsize=10)
    def _load_preview_page(self, page, dpi):
        if self.pdf_bytes is None:
            raise ValueError("PDF bytes are not loaded.")
        return load_page_bytes(self.pdf_bytes, page, dpi)

    def refresh_preview(self):
        """Refresh left preview panel."""
        if self.pdf_bytes is None:
            return
        try:
            page = int(self.page_spin.get())
        except Exception:
            page = 1
        dpi = int(self.dpi_scale.get())
        pdpi = max(30, dpi // 4)  # lighter preview DPI
        if self.split_spreads.get():
            base = (page + 1) // 2
            raw = load_page_bytes(self.pdf_bytes, base, pdpi)
            left, right = split_image_vertically(raw)
            img = left if page % 2 == 1 else right
        else:
            img = load_page_bytes(self.pdf_bytes, page, pdpi)

        self.current = img
        pre = self.preprocess_img(img)
        self._display(self.left_lbl, pre)
        self.status.config(text=f"Preview: page {page} @ {dpi} DPI (display {pdpi} DPI)")

    def _display(self, widget, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        widget.configure(image=img)
        widget.image = img

    def process_preview(self):
        """Manual trigger to re-run preprocessing and show preview."""
        if self.current is None:
            return
        pre = self.preprocess_img(self.current)
        self._display(self.left_lbl, pre)
        self.status.config(text="Processed preview")

    # --- CRAFT run ---

    def run_craft(self):
        """Prepare pages and run the CRAFT detector."""
        if self.pdf_bytes is None:
            messagebox.showwarning("No PDF", "Load a PDF first.")
            return

        # Disable UI & reset
        self.craft_btn.state(['disabled'])
        self.progress.configure(mode='determinate', value=0)
        self.craft_status.config(text="Preparing pages…")
        self.status.config(text="Preparing pages for CRAFT…")

        threading.Thread(target=self._run_craft_thread, daemon=True).start()

    def _run_craft_thread(self):
        import traceback
        try:
            processed_dir = PROCESSED_DIR
            result_dir    = RESULT_DIR
            craft_script  = Path(CRAFT_SCRIPT).resolve()
            craft_weights = Path(CRAFT_WEIGHTS).resolve()

            if not craft_script.exists():
                raise FileNotFoundError(f"CRAFT script not found:\n{craft_script}")
            if not craft_weights.exists():
                raise FileNotFoundError(f"CRAFT weights not found:\n{craft_weights}")

            # Clean & (re)create dirs
            shutil.rmtree(processed_dir, ignore_errors=True)
            shutil.rmtree(result_dir, ignore_errors=True)
            processed_dir.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)

            dpi = int(self.dpi_scale.get())
            total_pages = self.page_count * 2 if self.split_spreads.get() else self.page_count

            def set_progress(pct, text=None):
                self.parent.after(0, lambda: self.progress.configure(value=pct))
                if text is not None:
                    self.parent.after(0, lambda: self.craft_status.config(text=text))
                    self.parent.after(0, lambda: self.status.config(text=text))

            # Export preprocessed pages (always 3-channel)
            count = 0
            if self.split_spreads.get():
                for p in range(1, self.page_count + 1):
                    raw = load_page_bytes(self.pdf_bytes, p, dpi)
                    left, right = split_image_vertically(raw)
                    for half in (left, right):
                        pre = _ensure_3c(self.preprocess_img(half))
                        count += 1
                        out = processed_dir / f'page_{count:03d}.png'
                        ok = cv2.imwrite(str(out), pre)
                        if not ok:
                            raise IOError(f"Failed to write {out}")
                        set_progress(100 * count / max(1, total_pages), f"Exporting pages… ({count}/{total_pages})")
            else:
                for p in range(1, self.page_count + 1):
                    raw = load_page_bytes(self.pdf_bytes, p, dpi)
                    pre = _ensure_3c(self.preprocess_img(raw))
                    out = processed_dir / f'page_{p:03d}.png'
                    ok = cv2.imwrite(str(out), pre)
                    if not ok:
                        raise IOError(f"Failed to write {out}")
                    set_progress(100 * p / max(1, total_pages), f"Exporting pages… ({p}/{total_pages})")

            exported = sorted(processed_dir.glob('page_*.png'))
            if not exported:
                raise RuntimeError(f"No preprocessed pages found in {processed_dir}")
            unreadable = [str(p) for p in exported if (cv2.imread(str(p)) is None)]
            if unreadable:
                raise RuntimeError("Some images could not be read by OpenCV:\n" + "\n".join(unreadable[:20]))

            # Run CRAFT
            self.parent.after(0, lambda: self.progress.configure(mode='indeterminate'))
            self.parent.after(0, lambda: self.progress.start())
            set_progress(self.progress['value'], "Running CRAFT model…")

            proc = subprocess.run(
                [sys.executable, str(craft_script),
                 '--test_folder', str(processed_dir),
                 '--result_folder', str(result_dir),
                 '--trained_model', str(craft_weights)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(craft_script.parent)
            )

            self.parent.after(0, lambda: self.progress.stop())
            self.parent.after(0, lambda: self.progress.configure(mode='determinate', value=100))

            if proc.returncode != 0:
                err = (proc.stderr or "").strip()
                out = (proc.stdout or "").strip()
                msg = "CRAFT failed.\n\nSTDERR:\n{}\n\nSTDOUT:\n{}".format(err[:2000], out[:2000])
                self.parent.after(0, lambda: messagebox.showerror("CRAFT error", msg))
                return

            # Show first result image in right preview
            first_img = None
            for pat in ('res_page_*.jpg', 'res_page_*.png', 'page_*.png'):
                found = sorted(result_dir.glob(pat))
                if found:
                    first_img = found[0]
                    break
            if first_img is not None and first_img.exists():
                img = cv2.imread(str(first_img))
                if img is not None:
                    h, w = img.shape[:2]
                    img_small = cv2.resize(img, (w // 3, h // 3)) if max(h, w) > 1500 else img
                    self.parent.after(0, lambda: self._display(self.right_lbl, img_small))

            # Start transcript/page mapping → labeling flow
            # Start next stage depending on mode
            if self.var_use_onnx.get():
                self.parent.after(0, self.run_ocr_and_label)
            else:
                self.parent.after(0, self.start_pagebreak_wizard)
    
        except Exception as e:
            tb = traceback.format_exc()
            def _show():
                messagebox.showerror("Run CRAFT failed", f"{e}\n\n{tb}")
            self.parent.after(0, _show)
        finally:
            # Re-enable button
            self.parent.after(0, lambda: self.craft_btn.state(['!disabled']))

    # --- Transcript wizard & labeling flow ---

    def start_pagebreak_wizard(self):
        """Pick transcript file and map lines to pages via wizard."""
        transcript_path = filedialog.askopenfilename(
            title="Select full transcript (one line per text line)",
            filetypes=[("Text files", "*.txt")]
        )
        if not transcript_path:
            self.status.config(text="Transcript selection cancelled.")
            return
        with open(transcript_path, 'r', encoding='utf-8') as f:
            self.all_transcript_lines = [ln.rstrip('\n') for ln in f]
        page_txts = sorted(RESULT_DIR.glob('res_page_*.txt'))
        num_pages = len(page_txts)
        if num_pages == 0:
            messagebox.showinfo("No results", "No CRAFT results were found.")
            return

        def on_done(page_ranges):
            self.transcript_page_ranges = page_ranges
            self.open_labeling_sequence()

        PageBreakWizard(self.parent, self.all_transcript_lines, num_pages, on_done)

    def open_labeling_sequence(self):
        """Open per-page labelers in sequence."""
        triplets = find_result_triplets(RESULT_DIR)
        if not triplets:
            messagebox.showinfo("No results", "Unable to find result images/txt.")
            return
        self._labeling_triplets = triplets
        self._labeling_index = 0
        self.page_sessions = {}  # reset any previous
        self._open_next_labeler()

    def _open_next_labeler(self):
        """Open the next page's labeler or finish sequence."""
        if self._labeling_index >= len(self._labeling_triplets):
            # All pages visited; check approvals
            self._update_export_button_state()
            messagebox.showinfo("Labeling complete", "All pages visited. Approve remaining pages or export if all approved.")
            return

        img_path, txt_path, page_idx = self._labeling_triplets[self._labeling_index]
        boxes = parse_boxes(txt_path)
        line_groups = group_boxes_by_lines(boxes, y_thresh=self.cfg['line_y_thresh'])

        # Lines for this page from transcript ranges (0-based)
        page_lines = []
        if self.transcript_page_ranges and (0 <= page_idx - 1 < len(self.transcript_page_ranges)):
            start, end = self.transcript_page_ranges[page_idx - 1]  # res_page_001 -> index 0
            n = len(self.all_transcript_lines)
            if 0 <= start <= end < n:
                page_lines = self.all_transcript_lines[start:end + 1]

        # Restore existing draft if present
        if page_idx in self.page_sessions and 'lines' in self.page_sessions[page_idx]:
            saved = self.page_sessions[page_idx]['lines']
            page_lines = []
            for item in saved:
                if item.get('skipped', False):
                    page_lines.append("")
                else:
                    page_lines.append(" ".join(item.get('tokens', [])))

        def on_savedraft(cb_page_idx, page_data):
            self.page_sessions[cb_page_idx] = {
                'approved': self.page_sessions.get(cb_page_idx, {}).get('approved', False),
                'lines': page_data
            }
            self._update_export_button_state()

        def on_approve(cb_page_idx, page_data):
            self.page_sessions[cb_page_idx] = {'approved': True, 'lines': page_data}
            self._labeling_index += 1
            self._update_export_button_state()
            self._open_next_labeler()

        LineLabeler(self.parent, page_idx, img_path, line_groups, page_lines, on_approve, on_savedraft)

    def _update_export_button_state(self):
        """Enable Export only when every page in triplets is approved."""
        all_pages = [idx for (_img, _txt, idx) in self._labeling_triplets]
        if not all_pages:
            self.export_btn.state(['disabled'])
            return
        ok = all(self.page_sessions.get(i, {}).get('approved', False) for i in all_pages)
        if ok:
            self.export_btn.state(['!disabled'])
            self.status.config(text="All pages approved. Ready to export.")
        else:
            self.export_btn.state(['disabled'])
            approved = sum(1 for i in all_pages if self.page_sessions.get(i, {}).get('approved', False))
            self.status.config(text=f"Approved {approved}/{len(all_pages)} pages.")

    # --- Export to per-word class folders ---

    def export_dataset(self):
        """Export each approved word box as a rectangular PNG into class-named folders."""
        # Ensure all pages approved
        all_pages = [idx for (_img, _txt, idx) in self._labeling_triplets]
        if not all(self.page_sessions.get(i, {}).get('approved', False) for i in all_pages):
            messagebox.showerror("Cannot export", "All pages must be approved before exporting.")
            return

        # Prepare timestamped subfolder in data/data_6
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_root = DATASET_ROOT / f'export_{ts}'
        out_root.mkdir(parents=True, exist_ok=True)

        # Build export plan and count total words
        total_words = 0
        plan = []  # list of (page_idx, src_img_path, [(token, rect, line_idx, word_idx), ...])
        for (_overlay, _txt, page_idx) in self._labeling_triplets:
            sess = self.page_sessions.get(page_idx, {})
            if not sess.get('approved', False):
                continue
            # Crop from the CRAFT input page (preprocessed image); no extra processing per spec
            src_img = PROCESSED_DIR / f'page_{page_idx:03d}.png'
            img = cv2.imread(str(src_img))
            if img is None:
                continue  # skip silently if missing
            H, W = img.shape[:2]
            words_for_page = []
            for li, item in enumerate(sess.get('lines', [])):
                if item.get('skipped', False):
                    continue
                toks = item.get('tokens', [])
                boxes = item.get('boxes', [])
                if len(toks) != len(boxes):
                    continue  # should not happen after approval; safety guard
                for wi, (tok, quad) in enumerate(zip(toks, boxes)):
                    rect = rect_from_quad(quad, W, H)
                    if rect is None:
                        continue
                    words_for_page.append((tok, rect, li, wi))
            if words_for_page:
                plan.append((page_idx, src_img, words_for_page))
                total_words += len(words_for_page)

        if total_words == 0:
            messagebox.showinfo("Nothing to export", "No approved words found. Pages with no approved lines are skipped.")
            return

        # Export with progress
        self.progress.configure(mode='determinate', value=0, maximum=total_words)
        self.craft_status.config(text=f"Exporting {total_words} word crops…")
        exported = 0
        failed = 0

        for (page_idx, src_img, words) in plan:
            img = cv2.imread(str(src_img))
            if img is None:  # safety
                continue
            for (tok, rect, li, wi) in words:
                x0, y0, x1, y1 = rect
                crop = img[y0:y1, x0:x1]  # rectangular crop, no extra processing
                if crop.size == 0:
                    failed += 1
                    continue
                cls_dir = out_root / safe_class_dir(tok)
                cls_dir.mkdir(parents=True, exist_ok=True)
                fn = f"p{page_idx:03d}_l{li:03d}_w{wi:03d}.png"
                ok = cv2.imwrite(str(cls_dir / fn), crop)
                if not ok:
                    failed += 1
                exported += 1
                if exported % 25 == 0 or exported == total_words:
                    self.progress['value'] = exported
                    self.craft_status.config(text=f"Exported {exported}/{total_words}…")
                    self.parent.update_idletasks()

        messagebox.showinfo(
            "Export complete",
            f"Saved {exported - failed} word PNGs into:\n{out_root}\n\n"
            f"Class folders are the word strings (punctuation & case preserved).\n"
            f"{'Some crops failed to save.' if failed else 'All crops saved successfully.'}"
        )
        self.status.config(text=f"Exported {exported - failed} word images → {out_root}")


# === Run ===
if __name__ == '__main__':
    root = tk.Tk()
    app = PDFApp(root)
    app.pack(expand=True, fill='both')
    root.mainloop()
