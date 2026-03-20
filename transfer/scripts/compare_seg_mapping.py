"""Visual comparison of atlas120k masks remapped to open-h format.

Generates side-by-side comparison images:
  Left:   original RGB frame
  Center: atlas120k mask with original colors
  Right:  remapped open-h mask with open-h colors

Usage:
    python scripts/compare_seg_mapping.py                          # 5 random samples
    python scripts/compare_seg_mapping.py --num-samples 10         # 10 random samples
    python scripts/compare_seg_mapping.py --output-dir outputs/compare_masks
    python scripts/compare_seg_mapping.py --video 7RqTNr2Bmvk      # specific video only
    python scripts/compare_seg_mapping.py --all-classes             # pick frames maximizing class coverage
"""

import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# Atlas120k class definitions (from CLASS_MAPPING.txt)
# =============================================================================
ATLAS_CLASSES = {
    0: ("Background", (0, 0, 0)),
    1: ("Tools/camera", (255, 255, 255)),
    2: ("Vein (major)", (0, 0, 255)),
    3: ("Artery (major)", (255, 0, 0)),
    4: ("Nerve (major)", (255, 255, 0)),
    5: ("Small intestine", (0, 255, 0)),
    6: ("Colon/rectum", (0, 200, 100)),
    7: ("Abdominal wall", (200, 150, 100)),
    8: ("Diaphragm", (250, 150, 100)),
    9: ("Omentum", (255, 200, 100)),
    10: ("Aorta", (180, 0, 0)),
    11: ("Vena cava", (0, 0, 180)),
    12: ("Liver", (150, 100, 50)),
    13: ("Cystic duct", (0, 255, 255)),
    14: ("Gallbladder", (0, 200, 255)),
    15: ("Hepatic vein", (0, 100, 255)),
    16: ("Hepatic ligament", (255, 150, 50)),
    17: ("Cystic plate", (255, 220, 200)),
    18: ("Stomach", (200, 100, 200)),
    19: ("Ductus choledochus", (144, 238, 144)),
    20: ("Mesenterium", (247, 255, 0)),
    21: ("Ductus hepaticus", (255, 206, 27)),
    22: ("Spleen", (200, 0, 200)),
    41: ("Non-anatomical", (50, 50, 50)),
}

# =============================================================================
# Open-H class definitions
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

# =============================================================================
# Atlas120k -> Open-H mapping
# =============================================================================
ATLAS_TO_OPENH = {
    0: 0,   # Background -> Background
    1: 6,   # Tools/camera -> Instruments
    2: 7,   # Vein (major) -> Other Anatomy
    3: 7,   # Artery (major) -> Other Anatomy
    4: 7,   # Nerve (major) -> Other Anatomy
    5: 7,   # Small intestine -> Other Anatomy
    6: 7,   # Colon/rectum -> Other Anatomy
    7: 1,   # Abdominal wall -> Abdominal Wall
    8: 1,   # Diaphragm -> Abdominal Wall
    9: 4,   # Omentum -> Fat
    10: 7,  # Aorta -> Other Anatomy
    11: 7,  # Vena cava -> Other Anatomy
    12: 2,  # Liver -> Liver
    13: 5,  # Cystic duct -> Connective Tissue
    14: 3,  # Gallbladder -> Gallbladder
    15: 7,  # Hepatic vein -> Other Anatomy
    16: 5,  # Hepatic ligament -> Connective Tissue
    17: 5,  # Cystic plate -> Connective Tissue
    18: 7,  # Stomach -> Other Anatomy
    19: 5,  # Ductus choledochus -> Connective Tissue
    20: 5,  # Mesenterium -> Connective Tissue
    21: 5,  # Ductus hepaticus -> Connective Tissue
    22: 7,  # Spleen -> Other Anatomy
    41: 0,  # Non-anatomical -> Background
}


def remap_mask(mask_arr: np.ndarray) -> np.ndarray:
    """Remap atlas120k class IDs to open-h class IDs."""
    out = np.zeros_like(mask_arr)
    for atlas_id, openh_id in ATLAS_TO_OPENH.items():
        out[mask_arr == atlas_id] = openh_id
    return out


def colorize_mask(mask_arr: np.ndarray, palette: dict) -> np.ndarray:
    """Convert a class-ID mask to an RGB image using a palette dict {id: (name, (r,g,b))}."""
    h, w = mask_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, (_, color) in palette.items():
        rgb[mask_arr == cls_id] = color
    return rgb


def get_class_percentages(mask_arr: np.ndarray) -> dict:
    """Get {class_id: percentage} for classes present in mask."""
    total = mask_arr.size
    result = {}
    for v in np.unique(mask_arr):
        pct = np.sum(mask_arr == v) / total * 100
        if pct > 0.01:
            result[int(v)] = pct
    return result


def draw_legend(draw, x_start, y_start, classes_present, palette, font, max_h):
    """Draw a color legend on the given ImageDraw object. Returns height used."""
    y = y_start
    for cls_id in sorted(classes_present.keys()):
        if cls_id not in palette:
            continue
        name, color = palette[cls_id]
        pct = classes_present[cls_id]
        draw.rectangle([x_start, y, x_start + 16, y + 14], fill=color, outline=(80, 80, 80))
        label = f"{cls_id}: {name} ({pct:.1f}%)"
        draw.text((x_start + 22, y), label, fill=(220, 220, 220), font=font)
        y += 20
        if y > max_h - 10:
            break
    return y - y_start


def create_comparison(image_path: str, mask_path: str, output_path: str) -> dict:
    """Create a side-by-side comparison image. Returns class stats or empty dict on skip."""
    rgb = Image.open(image_path).convert("RGB")
    mask_img = Image.open(mask_path)

    if mask_img.mode == "RGB":
        print(f"  SKIP (RGB mask): {mask_path}")
        return {}

    mask_arr = np.array(mask_img)
    openh_arr = remap_mask(mask_arr)

    atlas_rgb = colorize_mask(mask_arr, ATLAS_CLASSES)
    openh_rgb = colorize_mask(openh_arr, OPENH_CLASSES)

    # Resize to common height
    target_h = 540
    aspect = rgb.width / rgb.height
    target_w = int(target_h * aspect)

    rgb_resized = rgb.resize((target_w, target_h), Image.LANCZOS)
    atlas_resized = Image.fromarray(atlas_rgb).resize((target_w, target_h), Image.NEAREST)
    openh_resized = Image.fromarray(openh_rgb).resize((target_w, target_h), Image.NEAREST)

    # Also create overlay: original with open-h mask at 40% opacity
    openh_overlay_arr = np.array(openh_resized).astype(np.float32)
    rgb_arr = np.array(rgb_resized).astype(np.float32)
    # Only overlay where mask is non-background
    openh_mask_resized = np.array(Image.fromarray(openh_arr.astype(np.uint8)).resize((target_w, target_h), Image.NEAREST))
    alpha = 0.45
    blended = rgb_arr.copy()
    fg = openh_mask_resized > 0
    blended[fg] = rgb_arr[fg] * (1 - alpha) + openh_overlay_arr[fg] * alpha
    overlay_resized = Image.fromarray(blended.clip(0, 255).astype(np.uint8))

    # Layout: 4 panels + legend
    gap = 3
    legend_w = 240
    panel_w = target_w
    total_w = panel_w * 4 + gap * 3 + legend_w
    header_h = 28
    total_h = target_h + header_h

    composite = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(composite)

    try:
        bold_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        bold_font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()
    try:
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except (OSError, IOError):
        small_font = ImageFont.load_default()

    labels = ["Original", "Atlas120k Classes", "Open-H Remapped", "Open-H Overlay"]
    panels = [rgb_resized, atlas_resized, openh_resized, overlay_resized]

    for i, (label, panel) in enumerate(zip(labels, panels)):
        x = i * (panel_w + gap)
        draw.text((x + panel_w // 2 - len(label) * 4, 6), label, fill=(200, 200, 220), font=bold_font)
        composite.paste(panel, (x, header_h))

    # Legend area (right side, split into atlas + open-h)
    lx = panel_w * 4 + gap * 3 + 8
    atlas_pcts = get_class_percentages(mask_arr)
    openh_pcts = get_class_percentages(openh_arr)

    draw.text((lx, 6), "Atlas120k", fill=(180, 180, 220), font=bold_font)
    draw_legend(draw, lx, 26, atlas_pcts, ATLAS_CLASSES, small_font, total_h // 2)

    mid_y = total_h // 2 + 10
    draw.text((lx, mid_y - 18), "Open-H", fill=(180, 180, 220), font=bold_font)
    draw_legend(draw, lx, mid_y, openh_pcts, OPENH_CLASSES, small_font, total_h)

    # Source path annotation
    parts = mask_path.split("/")
    # Find cholecystectomy/video/clip path
    try:
        idx = next(i for i, p in enumerate(parts) if p == "cholecystectomy")
        info_str = "/".join(parts[idx:])
    except StopIteration:
        info_str = "/".join(parts[-4:])
    draw.text((4, total_h - 14), info_str, fill=(100, 100, 100), font=small_font)

    composite.save(output_path, quality=95)
    return {"atlas": atlas_pcts, "openh": openh_pcts}


def find_diverse_samples(mask_files: list, num_samples: int) -> list:
    """Select frames that maximize class coverage via greedy set cover."""
    candidates = []
    for f in mask_files[::5]:
        img = Image.open(f)
        if img.mode == "RGB":
            continue
        m = np.array(img)
        classes = set(np.unique(m).tolist())
        candidates.append((f, classes))

    if not candidates:
        return []

    selected = []
    covered = set()
    remaining = list(candidates)

    for _ in range(num_samples):
        if not remaining:
            break
        best_idx = max(range(len(remaining)), key=lambda i: len(remaining[i][1] - covered))
        best = remaining.pop(best_idx)
        selected.append(best[0])
        covered.update(best[1])

    return selected


def main():
    parser = argparse.ArgumentParser(description="Compare atlas120k -> open-h segmentation class mapping")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of sample frames")
    parser.add_argument("--output-dir", type=str, default="outputs/compare_masks", help="Output directory")
    parser.add_argument("--video", type=str, default=None, help="Specific video ID to sample from")
    parser.add_argument("--all-classes", action="store_true", help="Pick frames maximizing class coverage")
    parser.add_argument("--data-root", type=str, default="../data/atlas120k", help="Path to atlas120k dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val/test)")
    args = parser.parse_args()

    data_dir = os.path.join(args.data_root, args.split, "cholecystectomy")
    if args.video:
        pattern = os.path.join(data_dir, args.video, "clip_*/masks/*.png")
    else:
        pattern = os.path.join(data_dir, "*/clip_*/masks/*.png")

    mask_files = sorted(glob.glob(pattern))
    print(f"Found {len(mask_files)} mask files in {data_dir}")

    if not mask_files:
        print("No mask files found. Check --data-root and --split.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Select samples
    if args.all_classes:
        samples = find_diverse_samples(mask_files, args.num_samples)
        print(f"Selected {len(samples)} diverse samples for max class coverage")
    else:
        valid = []
        for f in random.sample(mask_files, min(len(mask_files), args.num_samples * 20)):
            if Image.open(f).mode != "RGB":
                valid.append(f)
            if len(valid) >= args.num_samples:
                break
        samples = valid

    # Generate comparisons
    all_atlas_classes = set()
    all_openh_classes = set()
    generated = 0

    for i, mask_path in enumerate(samples):
        image_path = mask_path.replace("/masks/", "/images/").replace(".png", ".jpg")
        if not os.path.exists(image_path):
            print(f"  SKIP (no image): {image_path}")
            continue

        output_path = os.path.join(args.output_dir, f"compare_{i:03d}.jpg")
        stats = create_comparison(image_path, mask_path, output_path)

        if stats:
            all_atlas_classes.update(stats["atlas"].keys())
            all_openh_classes.update(stats["openh"].keys())
            generated += 1
            atlas_cls = sorted(stats["atlas"].keys())
            openh_cls = sorted(stats["openh"].keys())
            print(f"  [{generated}/{len(samples)}] {output_path}")
            print(f"    Atlas: {atlas_cls}  ->  Open-H: {openh_cls}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Generated {generated} comparisons in {args.output_dir}/")
    print(f"\nAll atlas classes seen: {sorted(all_atlas_classes)}")
    print(f"All open-h classes seen: {sorted(all_openh_classes)}")
    print(f"\nMapping summary:")
    for atlas_id in sorted(all_atlas_classes):
        if atlas_id in ATLAS_TO_OPENH:
            openh_id = ATLAS_TO_OPENH[atlas_id]
            atlas_name = ATLAS_CLASSES.get(atlas_id, ("?", (0, 0, 0)))[0]
            openh_name = OPENH_CLASSES[openh_id][0]
            print(f"  {atlas_id:3d} ({atlas_name:25s}) -> {openh_id} ({openh_name})")
        else:
            print(f"  {atlas_id:3d} (UNMAPPED)")


if __name__ == "__main__":
    main()
