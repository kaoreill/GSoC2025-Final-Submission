# GSoC2025-Final-Submission
## RenAIssance OCR  
**Improving Renaissance Spanish OCR with CNN–RNN Hybrids and Weighted Learning**  

Kate O’Reilly · Trinity College Dublin  
Google Summer of Code 2025  

---

## Overview  

This repository provides an end-to-end **OCR pipeline** for Renaissance-era Spanish manuscripts.  
The system integrates:  

- **Text detection**: CRAFT, Hi-SAM  
- **Text recognition**: CRNN (CNN → BiLSTM → CTC)  
- **Restoration**: DocumentEnhancer preprocessing & ShabbyPages-trained restoration model  
- **Data strategy**: synthetic + real training data with oversampling of rare characters  
- **Tooling**: data labelling web app, dual diplomatic & normalised transcription prototype  
- **Deployment**: ONNX export for portable inference  

---

## Features  

- **Detection**  
  - CRAFT for robust bounding box detection  
  - Hi-SAM for flexible region masks (curved/irregular lines)  

- **Recognition**  
  - CRNN trained with CTC Loss  
  - Beam search decoding with Spanish lexicon  
  - Weighted training for noisy data  
  - Export to **ONNX** for lightweight inference  

- **Restoration**  
  - *DocumentEnhancer*: grayscale → denoise → background flattening → CLAHE → binarisation → morphology  
  - *ShabbyPages Restorer*: UNet trained on [ShabbyPages dataset](https://github.com/sparkfish/shabby-pages) using L1 + MS-SSIM + gradient loss  

- **Data Strategy**  
  - Synthetic dataset built from Renaissance-style fonts (including ligatures & diacritics)  
  - Combined synthetic + real training for best generalisation  
  - **Synthetic oversampling** of rare characters (e.g. long-s, diacritics, abbreviations)  

- **Tooling**  
  - Web-based **data labelling app** for bounding boxes & transcriptions  
  - Prototype website for **dual transcriptions**:  
    - *Diplomatic*: faithful to manuscript spelling/abbreviations  
    - *Normalised*: expanded/modernised spelling for readability  

---

## Repository Structure  

```plaintext
RenAIssance_OCR/
│
├── detection/           # CRAFT, DBNet, PSENet, Hi-SAM configs & wrappers
├── recognition/         # CRNN training, decoding, ONNX export
├── restoration/         
│   ├── DocumentEnhancer.py   # Classical preprocessing pipeline
│   └── ShabbyPages/          # UNet restoration model + training scripts
├── data_labeller/       # App for annotation
├── website/             # Dual diplomatic + normalised transcription prototype
├── datasets/            # Links / scripts for real & synthetic data
└── README.md
```

## Training and Finetuning the CRNN

This section covers end-to-end training using `train_crnn_spanish_ocr.py` (the patched trainer for **CRNN (CNN → BiLSTM → CTC)**).

### TL;DR – Common Recipes

#### 1) Pretrain on **synthetic + rare** (with oversampling), then **fine-tune on real**
```bash
python train_crnn_spanish_ocr.py \
  --synthetic_dirs data/synth/ \
  --rare_dirs data/rare_chars/ \
  --real_dirs data/real/ \
  --out_dir runs/crnn_run01 \
  --pretrain_epochs 40 \
  --finetune_epochs 25 \
  --batch_size 64 \
  --height 40 \
  --scheduler onecycle \
  --early_stop_patience 5
```

#### 2) Finetune only (skip pretrain) using a checkpoint
```bash
python train_crnn_spanish_ocr.py \
  --real_dirs data/real/ \
  --out_dir runs/crnn_finetune_only \
  --finetune_only \
  --resume_from runs/crnn_run01/checkpoints/best_pretrain.pt \
  --finetune_epochs 25 \
  --batch_size 64
```

#### 3) Resume training (auto-handles changed charsets)
```bash
python train_crnn_spanish_ocr.py \
  --synthetic_dirs data/synth/ --rare_dirs data/rare_chars/ --real_dirs data/real/ \
  --out_dir runs/crnn_resume \
  --resume_from runs/crnn_run01/checkpoints/final.pt \
  --charset_mode union \
  --pretrain_epochs 10 --finetune_epochs 10
```

#### 4) Export ONNX at the end of training
```bash
python train_crnn_spanish_ocr.py \
  --synthetic_dirs data/synth/ --rare_dirs data/rare_chars/ --real_dirs data/real/ \
  --out_dir runs/crnn_run_onnx \
  --export_onnx runs/crnn_run_onnx/crnn.onnx
```

### Expected Data Layout

- Real: data/real/**/<word_image>.png (filenames are labels)
- Synthetic: data/synth/**/<word_image>.png
- Rare characters (oversampled): data/rare_chars/**/<word_image>.png

The script scans directories recursively; labels are extracted from filenames and cleaned (NFC normalisation, whitelist, trailing "(4)" removal, space collapse).

Some preprepared datasets can be found here: https://drive.google.com/drive/folders/1qxa7J-nKMwiyAysx3xKcBUGqFWUPE07f?usp=drive_link
Both a mix of synthetic and real

### Key Features

- Two-stage training
  - Pretrain on synthetic + rare with strong, realistic augmentations.
  - Fine-tune on real-only data with lighter augmentations and validation on real-only.

- Rare character support
  - Dedicated --rare_dirs that are aggressively sampled during pretrain.

- Label cleaning
  - NFC normalization, optional ftfy fix, whitelist, trailing counters removed.

- Dynamic widths
  - Default is keep-ratio + right-pad; use --fixed_width to force a fixed size.

- Schedulers
  - --scheduler onecycle (default) or --scheduler plateau (steps on CER).

- Early stopping
- --early_stop_patience N (0 disables).

- Resume + charset safety
  - --resume_from loads a checkpoint (handles DataParallel keys).
  - Final layer rows are remapped if the charset changes.
  - Choose charset policy with --charset_mode {resume|data|union|file}.

- Diagnostics
  - Saves up to --save_bad_preds_n mispredicted samples per validation into out_dir/bad_preds.
  - Writes charset.txt and charset.json in out_dir.
  - TensorBoard logs in out_dir/logs.

### Outputs
Inside --out_dir:
```plaintext
runs/crnn_run01/
├─ checkpoints/
│  ├─ best_pretrain.pt
│  ├─ best_finetune.pt
│  └─ final.pt
├─ logs/                # TensorBoard event files
├─ charset.txt
├─ charset.json
└─ bad_preds/           # optional: saved mispredictions (val)
```

## ShabbyPages Restorer: Training 

Train a UNet-style page **restoration** model on the paired **ShabbyPages** dataset (shabby/dirty ↔ clean/cleaned). The trainer supports explicit directory overrides and can also auto-pair mixed directories by filename tags (`*_shabby`, `*_cleaned`).

### Expected Dataset Layouts

```plaintext
ShabbyPages/
├─ train/
│ ├─ train/
│ │ ├─ train_shabby/ # degraded inputs
│ │ └─ train_cleaned/ # targets
└─ validate/
└─ validate/
├─ validate_shabby/
└─ validate_cleaned/
```

### Quickstart (Windows Config)
```bash
python restoration/train_shabbypages.py ^
  --train_dirty ".\ShabbyPages\train\train\train_shabby" ^
  --train_clean ".\ShabbyPages\train\train\train_cleaned" ^
  --val_dirty   ".\ShabbyPages\validate\validate\validate_shabby" ^
  --val_clean   ".\ShabbyPages\validate\validate\validate_cleaned" ^
  --epochs 25 --batch_size 4 --val_bs 2 ^
  --patch 256 --workers 0 ^
  --w_l1 1.0 --w_ms 0.0 --w_grad 0.0 --fg_weight 1.0 ^
  --cpu
```
### Recommended Training Schedules
- A. L1 warmup → add structure terms

```bash
# Warmup (L1 only)
python restoration/train_shabbypages.py --epochs 10 --w_l1 1.0 --w_ms 0.0 --w_grad 0.0 --patch 256

# Continue (load latest/best) with structure-aware loss
python restoration/train_shabbypages.py --epochs 30 --w_l1 1.0 --w_ms 0.2 --w_grad 0.2 --patch 256
```

- B. Larger patches for more context
```bash
python restoration/train_shabbypages.py --epochs 20 --batch_size 4 --patch 384
```

### Advanced Options
- --extra_synth : injects JPEG + spot noise on inputs for robustness.
- --val_patch : larger validation tiles (e.g., 768) to better reflect page-scale behavior.
- --base : UNet width multiplier (default 48; increase if you have more VRAM).
- --lr : learning rate (default 2e-4 with AdamW + cosine anneal).
- --ckpt : resume from checkpoint (restores model/optim/scheduler state).

### Outputs
A typical run directory
```plaintext
runs/YYYYMMDD_HHMMSS/
├─ ckpt/
│  ├─ latest.pt
│  └─ best.pt
├─ samples/
│  ├─ train_0001000.png       # periodic grids (input | pred | target)
│  └─ val_0001200_003.png     # validation triptychs
└─ logs (console)
```

## Tools
#### Data Labelling App

Run locally 

```bash
cd data_labeller
python app.py
```

#### Dual Transceiption Website Prototype

Displays page images with bounding box overlays and side-by-side diplomatic vs normalised transcription.

## License

This project is licensed under the MIT License. See the LICENSE file
