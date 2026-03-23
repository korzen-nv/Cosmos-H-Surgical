"""Visual comparison of segmentation masks from all 4 datasets remapped to Open-H format.

Generates comparison grids: original image | original mask (dataset colors) | Open-H remapped mask
16 random samples per dataset, saved to outputs/compare_masks_unified/.

Usage:
    cd transfer
    python scripts/compare_all_datasets_seg.py
    python scripts/compare_all_datasets_seg.py --num-samples 8
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# Open-H target classes
# =============================================================================
OPENH_CLASSES = {
    0: ("Background", (0, 0, 0)),
    1: ("Abdominal Wall", (64, 64, 64)),
    2: ("Liver", (255, 73, 40)),
    3: ("Gallbladder", (255, 45, 255)),
    4: ("Fat", (57, 4, 105)),
    5: ("Connective Tissue", (137, 126, 8)),
    6: ("Instruments", (100, 228, 3)),
    7: ("Other Anatomy", (74, 74, 138)),
}

OPENH_COLOR_LUT = np.zeros((256, 3), dtype=np.uint8)
for cid, (_, rgb) in OPENH_CLASSES.items():
    OPENH_COLOR_LUT[cid] = rgb


# =============================================================================
# Atlas120k
# =============================================================================
ATLAS_PALETTE = {
    0: ("Background", (0, 0, 0)),
    1: ("Tools/camera", (255, 255, 255)),
    7: ("Abdominal wall", (200, 150, 100)),
    9: ("Omentum", (255, 200, 100)),
    12: ("Liver", (150, 100, 50)),
    13: ("Cystic duct", (0, 255, 255)),
    14: ("Gallbladder", (0, 200, 255)),
    16: ("Hepatic ligament", (255, 150, 50)),
    17: ("Cystic plate", (255, 220, 200)),
    18: ("Stomach", (200, 100, 200)),
    41: ("Non-anatomical", (50, 50, 50)),
}

ATLAS_TO_OPENH = {
    0: 0, 1: 6, 2: 7, 3: 7, 4: 7, 5: 7, 6: 7, 7: 1, 8: 1, 9: 4,
    10: 7, 11: 7, 12: 2, 13: 5, 14: 3, 15: 7, 16: 5, 17: 5, 18: 7,
    19: 5, 20: 5, 21: 5, 22: 7, 41: 0, 43: 7, 44: 7, 45: 7, 46: 7,
}

ATLAS_REMAP_LUT = np.zeros(256, dtype=np.uint8)
for src, dst in ATLAS_TO_OPENH.items():
    ATLAS_REMAP_LUT[src] = dst


# =============================================================================
# CholecSeg8k — use *_endo_color_mask.png (filled RGB masks)
# Map actual RGB colors to Open-H classes via watershed cross-reference.
# =============================================================================
CHOLECSEG_COLOR_MAP = {
    # RGB color in color_mask -> (class_name, open_h_id)
    (127, 127, 127): ("Black Background", 0),
    (210, 140, 140): ("Abdominal Wall", 1),
    (255, 114, 114): ("Liver", 2),
    (231, 70, 156):  ("GI Tract", 7),
    (186, 183, 75):  ("Fat", 4),
    (170, 255, 0):   ("Grasper", 6),
    (255, 85, 0):    ("Connective Tissue", 5),
    (255, 0, 0):     ("Blood", 7),
    (255, 255, 0):   ("Cystic Duct", 5),
    (169, 255, 184): ("L-hook Electrocautery", 6),
    (255, 160, 165): ("Gallbladder", 3),
    (0, 50, 128):    ("Hepatic Vein", 7),
    (111, 74, 0):    ("Liver Ligament", 5),
    (255, 255, 255): ("Boundary", 0),
    (0, 0, 0):       ("Background", 0),
}


# =============================================================================
# Endoscapes — grayscale masks, class ID = pixel value
# =============================================================================
ENDOSCAPES_PALETTE = {
    0: ("background", (40, 40, 40)),
    1: ("cystic_plate", (137, 126, 8)),
    2: ("calot_triangle", (180, 160, 20)),
    3: ("cystic_artery", (255, 0, 0)),
    4: ("cystic_duct", (0, 255, 255)),
    5: ("gallbladder", (255, 45, 255)),
    6: ("tool", (100, 228, 3)),
}

ENDOSCAPES_TO_OPENH = {0: 0, 1: 5, 2: 5, 3: 7, 4: 5, 5: 3, 6: 6, 255: 0}

ENDOSCAPES_REMAP_LUT = np.zeros(256, dtype=np.uint8)
for src, dst in ENDOSCAPES_TO_OPENH.items():
    ENDOSCAPES_REMAP_LUT[src] = dst


# =============================================================================
# HeiSurf — RGB color-encoded masks
# =============================================================================
HEISURF_COLOR_TO_CLASS = {
    (253, 101, 14): ("Abdominal Wall", 1),
    (41, 10, 12): ("Blood Pool", 7),
    (62, 62, 62): ("Censored", 0),
    (255, 51, 0): ("Clip", 6),
    (51, 221, 255): ("Cystic Artery", 7),
    (153, 255, 51): ("Cystic Duct", 5),
    (1, 68, 34): ("Drainage", 6),
    (255, 255, 0): ("Fatty Tissue", 4),
    (25, 176, 34): ("Gallbladder", 3),
    (0, 55, 104): ("GB Resection Bed", 5),
    (74, 165, 255): ("GI Tract", 7),
    (255, 255, 255): ("Gauze", 0),
    (255, 168, 0): ("Hilum/Hepatoduodenal", 5),
    (160, 56, 133): ("Inside Trocar", 0),
    (101, 46, 151): ("Instrument", 6),
    (136, 0, 0): ("Liver", 2),
    (204, 204, 204): ("Other", 7),
    (82, 82, 82): ("Out of Image", 0),
    (61, 245, 61): ("Round/Falciform Lig.", 5),
    (79, 48, 31): ("Specimenbag", 0),
    (255, 51, 119): ("Trocar", 0),
    (240, 240, 240): ("Uncertainties", 0),
    (0, 0, 0): ("Background", 0),
}


# =============================================================================
# Utility functions
# =============================================================================
def colorize_openh(mask_ids: np.ndarray) -> np.ndarray:
    return OPENH_COLOR_LUT[mask_ids]


def get_class_stats(mask_ids: np.ndarray, palette: dict) -> dict:
    total = mask_ids.size
    stats = {}
    for v in np.unique(mask_ids):
        pct = np.sum(mask_ids == v) / total * 100
        if pct > 0.05:
            stats[int(v)] = pct
    return stats


def draw_legend(draw, x, y, stats, palette, font, max_h):
    for cid in sorted(stats.keys()):
        if cid in palette:
            name, color = palette[cid]
        else:
            name, color = f"class_{cid}", (128, 128, 128)
        draw.rectangle([x, y, x + 14, y + 12], fill=color, outline=(80, 80, 80))
        draw.text((x + 18, y), f"{cid}: {name} ({stats[cid]:.1f}%)", fill=(200, 200, 200), font=font)
        y += 16
        if y > max_h - 10:
            break


def create_comparison(image: np.ndarray, orig_mask_rgb: np.ndarray, openh_ids: np.ndarray,
                      orig_stats: dict, openh_stats: dict, orig_palette: dict,
                      source_label: str, sample_info: str, output_path: str):
    """Create a 4-panel comparison image."""
    h_target = 360
    aspect = image.shape[1] / image.shape[0]
    w_target = int(h_target * aspect)

    img_r = cv2.resize(image, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
    orig_r = cv2.resize(orig_mask_rgb, (w_target, h_target), interpolation=cv2.INTER_NEAREST)
    openh_rgb = colorize_openh(openh_ids)
    openh_r = cv2.resize(openh_rgb, (w_target, h_target), interpolation=cv2.INTER_NEAREST)

    # Overlay
    overlay = img_r.astype(np.float32).copy()
    fg = openh_ids != 0
    fg_r = cv2.resize(fg.astype(np.uint8), (w_target, h_target), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay[fg_r] = img_r[fg_r] * 0.55 + openh_r[fg_r].astype(np.float32) * 0.45
    overlay = overlay.clip(0, 255).astype(np.uint8)

    gap = 2
    legend_w = 200
    pw = w_target
    header_h = 22
    total_w = pw * 4 + gap * 3 + legend_w
    total_h = h_target + header_h

    comp = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(comp)

    try:
        bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except (OSError, IOError):
        bold = small = ImageFont.load_default()

    labels = ["Original", f"{source_label} Mask", "Open-H Remap", "Overlay"]
    panels = [img_r, orig_r, openh_r, overlay]
    for i, (label, panel) in enumerate(zip(labels, panels)):
        x = i * (pw + gap)
        draw.text((x + pw // 2 - len(label) * 3, 4), label, fill=(180, 180, 200), font=bold)
        comp.paste(Image.fromarray(panel), (x, header_h))

    # Legends
    lx = pw * 4 + gap * 3 + 4
    draw.text((lx, 4), source_label, fill=(160, 160, 200), font=bold)
    draw_legend(draw, lx, 20, orig_stats, orig_palette, small, total_h // 2)
    mid_y = total_h // 2 + 8
    draw.text((lx, mid_y - 14), "Open-H", fill=(160, 160, 200), font=bold)
    draw_legend(draw, lx, mid_y, openh_stats, OPENH_CLASSES, small, total_h)

    draw.text((4, total_h - 12), sample_info, fill=(100, 100, 100), font=small)
    comp.save(output_path, quality=95)


# =============================================================================
# Dataset-specific sample generators
# =============================================================================
def sample_atlas120k(data_root, num_samples):
    """Sample from Atlas120k cholecystectomy."""
    mask_files = sorted(glob.glob(os.path.join(data_root, "train/cholecystectomy/*/clip_*/masks/*.png")))
    valid = [f for f in mask_files if Image.open(f).mode != "RGB"][:num_samples * 5]
    samples = random.sample(valid, min(num_samples, len(valid)))

    results = []
    for mask_path in samples:
        img_path = mask_path.replace("/masks/", "/images/").replace(".png", ".jpg")
        if not os.path.exists(img_path):
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # Original colorized mask
        orig_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cid, (_, color) in ATLAS_PALETTE.items():
            orig_rgb[mask == cid] = color

        openh_ids = ATLAS_REMAP_LUT[mask]
        orig_stats = get_class_stats(mask, ATLAS_PALETTE)
        openh_stats = get_class_stats(openh_ids, OPENH_CLASSES)

        info = "/".join(mask_path.split("/")[-4:])
        results.append((image, orig_rgb, openh_ids, orig_stats, openh_stats, ATLAS_PALETTE, "Atlas120k", info))
    return results


def sample_cholecseg8k(data_root, num_samples):
    """Sample from CholecSeg8k using filled color masks."""
    mask_files = sorted(glob.glob(os.path.join(data_root, "video*/video*/*_endo_color_mask.png")))
    samples = random.sample(mask_files, min(num_samples, len(mask_files)))

    results = []
    for mask_path in samples:
        img_path = mask_path.replace("_endo_color_mask.png", "_endo.png")
        if not os.path.exists(img_path):
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        h, w, _ = mask_rgb.shape

        # Map RGB colors to Open-H class IDs
        openh_ids = np.zeros((h, w), dtype=np.uint8)
        for color, (name, openh_id) in CHOLECSEG_COLOR_MAP.items():
            r, g, b = color
            match = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
            openh_ids[match] = openh_id

        # Build legend from actual colors present
        cs8k_legend = {}
        idx = 0
        for color, (name, _) in CHOLECSEG_COLOR_MAP.items():
            r, g, b = color
            match = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
            pct = match.sum() / (h * w) * 100
            if pct > 0.05:
                cs8k_legend[idx] = pct
                idx += 1

        cs8k_palette = {}
        idx = 0
        for color, (name, _) in CHOLECSEG_COLOR_MAP.items():
            r, g, b = color
            match = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
            pct = match.sum() / (h * w) * 100
            if pct > 0.05:
                cs8k_palette[idx] = (name, color)
                idx += 1

        openh_stats = get_class_stats(openh_ids, OPENH_CLASSES)
        info = "/".join(mask_path.split("/")[-3:])
        results.append((image, mask_rgb, openh_ids, cs8k_legend, openh_stats,
                         cs8k_palette, "CholecSeg8k", info))
    return results


def sample_endoscapes(data_root, num_samples):
    """Sample from Endoscapes."""
    mask_files = sorted(glob.glob(os.path.join(data_root, "semseg/*.png")))
    samples = random.sample(mask_files, min(num_samples, len(mask_files)))

    results = []
    for mask_path in samples:
        # Image files are in train/val/test dirs or train_seg/val_seg/test_seg
        fname = os.path.basename(mask_path).replace(".png", ".jpg")
        img_path = None
        for subdir in ["train", "val", "test"]:
            candidate = os.path.join(data_root, subdir, fname)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # Original colorized
        orig_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cid, (_, color) in ENDOSCAPES_PALETTE.items():
            orig_rgb[mask == cid] = color

        openh_ids = ENDOSCAPES_REMAP_LUT[mask]
        orig_stats = get_class_stats(mask, ENDOSCAPES_PALETTE)
        openh_stats = get_class_stats(openh_ids, OPENH_CLASSES)
        info = os.path.basename(mask_path)
        results.append((image, orig_rgb, openh_ids, orig_stats, openh_stats, ENDOSCAPES_PALETTE, "Endoscapes", info))
    return results


def sample_heisurf(data_root, num_samples):
    """Sample from HeiSurf."""
    mask_files = sorted(glob.glob(os.path.join(data_root, "Segmentation/*/*.png")))
    samples = random.sample(mask_files, min(num_samples, len(mask_files)))

    results = []
    for mask_path in samples:
        # Image path: Frames/{case_id}/same_filename
        parts = mask_path.split("/")
        case_id = parts[-2]
        fname = parts[-1]
        img_path = os.path.join(data_root, "Frames", case_id, fname)
        if not os.path.exists(img_path):
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        # Map RGB colors to Open-H IDs
        h, w, _ = mask_rgb.shape
        openh_ids = np.zeros((h, w), dtype=np.uint8)
        orig_display = mask_rgb.copy()  # use original colors for display

        # Build a quick lookup
        for color, (name, openh_id) in HEISURF_COLOR_TO_CLASS.items():
            r, g, b = color
            match = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
            openh_ids[match] = openh_id

        # Stats using original RGB as proxy for class identity
        orig_stats = {}
        for color, (name, _) in HEISURF_COLOR_TO_CLASS.items():
            r, g, b = color
            match = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
            pct = match.sum() / (h * w) * 100
            if pct > 0.05:
                orig_stats[name] = pct

        # Convert orig_stats format for legend
        heisurf_legend = {}
        idx = 0
        for name, pct in sorted(orig_stats.items(), key=lambda x: -x[1]):
            # Find the color for this class
            for color, (n, _) in HEISURF_COLOR_TO_CLASS.items():
                if n == name:
                    heisurf_legend[idx] = pct
                    break
            idx += 1

        # For legend, use a simple palette mapping
        heisurf_display_palette = {}
        idx = 0
        for name, pct in sorted(orig_stats.items(), key=lambda x: -x[1]):
            for color, (n, _) in HEISURF_COLOR_TO_CLASS.items():
                if n == name:
                    heisurf_display_palette[idx] = (name, color)
                    break
            idx += 1

        openh_stats = get_class_stats(openh_ids, OPENH_CLASSES)
        info = f"{case_id}/{fname}"
        results.append((image, orig_display, openh_ids, heisurf_legend, openh_stats,
                         heisurf_display_palette, "HeiSurf", info))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs/compare_masks_unified")
    parser.add_argument("--data-root", default="../data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(42)

    datasets = [
        ("atlas120k", sample_atlas120k, os.path.join(args.data_root, "atlas120k")),
        ("cholecseg8k", sample_cholecseg8k, os.path.join(args.data_root, "cholecSeg8k")),
        ("endoscapes", sample_endoscapes, os.path.join(args.data_root, "endoscapes")),
        ("heisurf", sample_heisurf, os.path.join(args.data_root, "HeiSurf")),
    ]

    for ds_name, sample_fn, ds_path in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name} ({ds_path})")

        if not os.path.isdir(ds_path):
            print(f"  NOT FOUND — skipping")
            continue

        samples = sample_fn(ds_path, args.num_samples)
        print(f"  {len(samples)} samples")

        for i, (image, orig_rgb, openh_ids, orig_stats, openh_stats, palette, label, info) in enumerate(samples):
            out_path = os.path.join(args.output_dir, f"{ds_name}_{i:02d}.jpg")
            create_comparison(image, orig_rgb, openh_ids, orig_stats, openh_stats, palette, label, info, out_path)
            print(f"  [{i+1}/{len(samples)}] {out_path}")

    print(f"\nAll comparisons saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
