# Datasets

## Segmentation Training Datasets

Four cholecystectomy segmentation datasets are used to train the Open-H segmentation controlnet.
All are mapped to a unified 8-class Open-H format (see [`MAPPINGS.md`](MAPPINGS.md) for details).

### Summary

| Dataset | Clips | Orig Frames | Resolution | Mask Format | License |
|---------|-------|-------------|------------|-------------|---------|
| Atlas120k (cholec only) | 95 | 26,259 | 1920x1080 | Indexed PNG (P-mode) | CC BY-NC-SA 4.0 |
| CholecSeg8k | 101 | 8,080 | 854x480 | RGB color mask PNG | CC BY-NC-SA 4.0 |
| Endoscapes | 7 | 183 | 854x480 | Grayscale PNG | CC BY-NC-SA 4.0 |
| HeiSurf | 8 | 228 | 1920x1080 | RGB color mask PNG | CC BY-NC-SA 4.0 |
| **Total** | **211** | **34,432** | | | |

### Processed (Open-H Unified)

All datasets are converted to `<name>.mp4` + `<name>.seg.mp4` pairs at 1280-wide resolution,
16 fps. Clips shorter than 93 frames are padded by repeating the last frame.

| Location | Path |
|----------|------|
| Local | `data/atlas120k_openh/` (Atlas120k only) |
| Cluster | `/lustre/.../cosmos/datasets/openh_unified/` (all 4 datasets) |

**Output format** (per video):
- `<name>.mp4` — RGB video, H.264 CRF 18
- `<name>.seg.mp4` — Open-H colored segmentation, lossless H.264 (CRF 0, yuv444p)
- `<name>.json` — Caption `{"short": "A laparoscopic cholecystectomy..."}`
- `train.json` — Index `{"training": [{"video": "name.mp4"}, ...]}`

### Preparation Scripts

```bash
# Atlas120k only (local or cluster)
python scripts/prepare_atlas120k_openh.py --data-root ../data/atlas120k --output-root data/atlas120k_openh

# All 4 datasets unified (cluster)
sbatch prepare_unified_data_mars.slurm

# Or manually:
python scripts/prepare_openh_unified.py \
    --output-root /path/to/output \
    --atlas120k /path/to/atlas120k \
    --cholecseg8k /path/to/cholecSeg8k \
    --endoscapes /path/to/endoscapes \
    --heisurf /path/to/HeiSurf \
    --min-frames 20 --target-frames 93 --width 1280
```

---

## Dataset Details

### Atlas120k

- **Source:** ATLAS-120k surgical video dataset
- **Procedure:** Cholecystectomy (17 videos, 95 clips across train/val/test)
- **Annotations:** 47 anatomical classes, hand-annotated per-frame masks
- **Mask encoding:** Indexed-color PNG (PIL mode=P), pixel value = class ID
- **Caveats:**
  - Some clips have RGB-mode masks (non-standard encoding) — these are skipped
  - Only cholecystectomy subset is used (14 surgical types available)
- **Local path:** `/data/atlas120k/` (symlink to `/home/pkorzeniowsk/Datasets/atlas120k/`)
- **Cluster path:** `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/atlas120k/atlas120k/`

### CholecSeg8k

- **Source:** CholecSeg8k dataset (Cholec80 subset)
- **Procedure:** Cholecystectomy (17 videos, 101 frame groups of 80 frames each)
- **Annotations:** 13 classes, filled color masks
- **Mask encoding:** RGB PNG (`*_endo_color_mask.png`). Each class has a unique RGB color.
  The `*_endo_mask.png` files are contour-only (not filled) — do not use for training.
- **Pixel value encoding:** The documented hex codes (e.g., #505050) encode pixel values as
  the two-digit hex prefix read as plain decimal: #505050 → pixel value 50 in `_endo_mask.png`.
  Use `_endo_color_mask.png` for filled masks with actual RGB colors.
- **Local path:** `/data/cholecSeg8k/`
- **Cluster path:** `/lustre/.../cosmos/datasets/cholecseg8k/`

### Endoscapes

- **Source:** Endoscapes dataset
- **Procedure:** Cholecystectomy (50 videos, 493 frames total — very sparse)
- **Annotations:** 7 classes focused on Critical View of Safety (CVS) structures
- **Mask encoding:** Grayscale PNG (`semseg/*.png`), pixel value = class ID (0-6, 255=boundary)
- **Note:** Only 7 clips have ≥20 frames. Heavy padding needed (up to +72 frames).
  No liver, abdominal wall, or fat annotations — these map to Background.
- **Local path:** `/data/endoscapes/`
- **Cluster path:** `/lustre/.../cosmos/datasets/endoscapes/endoscapes/`

### HeiSurf

- **Source:** HeiSurf (Heidelberg Surgical Fine-grained) dataset
- **Procedure:** Cholecystectomy (24 cases, 466 frames total — sparse, step=2)
- **Annotations:** 22 classes including non-anatomy (gauze, trocar, specimenbag, etc.)
- **Mask encoding:** RGB PNG (`Segmentation/{case_id}/*.png`), each class = unique RGB color.
  Images in `Frames/{case_id}/*.png`.
- **Note:** Only 8 cases have ≥20 frames. Heavy padding needed. Rich annotation includes
  clips, drainage, gauze, trocar — all mapped to Background in Open-H.
- **Local path:** `/data/HeiSurf/`
- **Cluster path:** `/lustre/.../cosmos/datasets/HeiSurf/HeiSurf/`

---

## Open-H Inference Dataset

Used for inference evaluation, not training. See CLAUDE.md for details.

- **Path:** `inputs/open_h/<session>/`
- **Source:** Open-H Non-Expert recordings with color, depth, and segmentation videos
- **Resolution:** 720p (1280x720)

---

## Cluster Paths Reference

```
/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk/cosmos/
├── datasets/
│   ├── atlas120k_openh/          # Atlas120k only (first training run)
│   │   ├── train/                # 25 videos
│   │   ├── val/                  # 16 videos
│   │   ├── train.json
│   │   └── val.json
│   ├── openh_unified/            # All 4 datasets combined
│   │   ├── train/                # 211 videos
│   │   └── train.json
│   ├── cholecseg8k/              # Raw CholecSeg8k
│   ├── endoscapes/endoscapes/    # Raw Endoscapes
│   └── HeiSurf/HeiSurf/         # Raw HeiSurf
├── checkpoints/
│   ├── cosmos-h-surgical-transfer-seg_model_ema_bf16.pt  # Original HF seg checkpoint
│   ├── seg_openh_model_ema_bf16.pt                       # Fine-tuned (atlas120k, 600 iters)
│   └── output/.../checkpoints/iter_000000{100-600}/      # DCP training checkpoints
└── Cosmos-H-Surgical/transfer/   # Git repo working copy
```

`/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/`
contains the original Atlas120k dataset (read-only shared storage).
