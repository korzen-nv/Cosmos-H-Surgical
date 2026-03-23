"""Prepare unified Open-H training data from 4 cholecystectomy segmentation datasets.

Converts Atlas120k, CholecSeg8k, Endoscapes, and HeiSurf into the format expected
by SingleViewTransferDatasetJSON:
    <name>.mp4       — RGB video at 1280xH
    <name>.seg.mp4   — Open-H colored segmentation video at 1280xH
    <name>.json      — Caption {"short": "..."}
    train.json       — Index {"training": [{"video": "..."}]}

Short clips are padded to min_video_frames by repeating the last frame.

Usage:
    python scripts/prepare_openh_unified.py --output-root /path/to/output
    python scripts/prepare_openh_unified.py --dry-run
    python scripts/prepare_openh_unified.py --min-frames 20 --target-frames 93
"""

import argparse
import glob
import json
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# =============================================================================
# Open-H class palette (8 classes)
# =============================================================================
OPENH_COLOR_LUT = np.zeros((256, 3), dtype=np.uint8)
OPENH_COLOR_LUT[0] = (0, 0, 0)        # Background
OPENH_COLOR_LUT[1] = (64, 64, 64)     # Abdominal Wall
OPENH_COLOR_LUT[2] = (255, 73, 40)    # Liver
OPENH_COLOR_LUT[3] = (255, 45, 255)   # Gallbladder
OPENH_COLOR_LUT[4] = (57, 4, 105)     # Fat
OPENH_COLOR_LUT[5] = (137, 126, 8)    # Connective Tissue
OPENH_COLOR_LUT[6] = (100, 228, 3)    # Instruments
OPENH_COLOR_LUT[7] = (74, 74, 138)    # Other Anatomy

DEFAULT_CAPTION = (
    "A laparoscopic cholecystectomy surgical video showing the "
    "dissection and removal of the gallbladder from the liver bed. "
    "Surgical instruments manipulate tissue in the hepatocystic triangle "
    "region, with liver, gallbladder, fat, and connective tissue visible."
)

# =============================================================================
# Atlas120k: indexed PNG masks → Open-H
# =============================================================================
ATLAS_REMAP_LUT = np.zeros(256, dtype=np.uint8)
for src, dst in {
    0: 0, 1: 6, 2: 7, 3: 7, 4: 7, 5: 7, 6: 7, 7: 1, 8: 1, 9: 4,
    10: 7, 11: 7, 12: 2, 13: 5, 14: 3, 15: 7, 16: 5, 17: 5, 18: 7,
    19: 5, 20: 5, 21: 5, 22: 7, 41: 0, 43: 7, 44: 7, 45: 7, 46: 7,
}.items():
    ATLAS_REMAP_LUT[src] = dst

# =============================================================================
# CholecSeg8k: color mask RGB → Open-H
# =============================================================================
CHOLECSEG_COLOR_MAP = {
    (127, 127, 127): 0,  # Black Background
    (210, 140, 140): 1,  # Abdominal Wall
    (255, 114, 114): 2,  # Liver
    (231, 70, 156): 7,   # GI Tract → Other Anatomy
    (186, 183, 75): 4,   # Fat
    (170, 255, 0): 6,    # Grasper → Instruments
    (255, 85, 0): 5,     # Connective Tissue
    (255, 0, 0): 7,      # Blood → Other Anatomy
    (255, 255, 0): 5,    # Cystic Duct → Connective Tissue
    (169, 255, 184): 6,  # L-hook → Instruments
    (255, 160, 165): 3,  # Gallbladder
    (0, 50, 128): 7,     # Hepatic Vein → Other Anatomy
    (111, 74, 0): 5,     # Liver Ligament → Connective Tissue
    (255, 255, 255): 0,  # Boundary → Background
    (0, 0, 0): 0,        # Background
}

# =============================================================================
# Endoscapes: grayscale mask class ID → Open-H
# =============================================================================
ENDOSCAPES_REMAP_LUT = np.zeros(256, dtype=np.uint8)
for src, dst in {0: 0, 1: 5, 2: 5, 3: 7, 4: 5, 5: 3, 6: 6, 255: 0}.items():
    ENDOSCAPES_REMAP_LUT[src] = dst

# =============================================================================
# HeiSurf: RGB color → Open-H
# =============================================================================
HEISURF_COLOR_MAP = {
    (253, 101, 14): 1,   # Abdominal Wall
    (41, 10, 12): 7,     # Blood Pool → Other Anatomy
    (62, 62, 62): 0,     # Censored → Background
    (255, 51, 0): 6,     # Clip → Instruments
    (51, 221, 255): 7,   # Cystic Artery → Other Anatomy
    (153, 255, 51): 5,   # Cystic Duct → Connective Tissue
    (1, 68, 34): 6,      # Drainage → Instruments
    (255, 255, 0): 4,    # Fatty Tissue → Fat
    (25, 176, 34): 3,    # Gallbladder
    (0, 55, 104): 5,     # GB Resection Bed → Connective Tissue
    (74, 165, 255): 7,   # GI Tract → Other Anatomy
    (255, 255, 255): 0,  # Gauze → Background
    (255, 168, 0): 5,    # Hilum/Hepatoduodenal → Connective Tissue
    (160, 56, 133): 0,   # Inside Trocar → Background
    (101, 46, 151): 6,   # Instrument
    (136, 0, 0): 2,      # Liver
    (204, 204, 204): 7,  # Other → Other Anatomy
    (82, 82, 82): 0,     # Out of Image → Background
    (61, 245, 61): 5,    # Round/Falciform Lig. → Connective Tissue
    (79, 48, 31): 0,     # Specimenbag → Background
    (255, 51, 119): 0,   # Trocar → Background
    (240, 240, 240): 0,  # Uncertainties → Background
    (0, 0, 0): 0,        # Background
}


# =============================================================================
# Common utilities
# =============================================================================
def colorize_openh(class_ids: np.ndarray) -> np.ndarray:
    """Convert Open-H class ID mask to RGB. Input: (H,W) uint8. Output: (H,W,3) uint8."""
    return OPENH_COLOR_LUT[class_ids]


def remap_rgb_mask(mask_rgb: np.ndarray, color_map: dict) -> np.ndarray:
    """Map RGB color mask to Open-H class IDs using a color→class dict."""
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for color, openh_id in color_map.items():
        r, g, b = color
        match = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
        out[match] = openh_id
    return out


def encode_video_cv2(frames: list[np.ndarray], output_path: str, fps: int, lossless: bool = False):
    """Encode RGB frames to H.264 MP4."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    # Re-encode to H.264 for decord compatibility
    tmp = output_path + ".tmp.mp4"
    os.rename(output_path, tmp)
    if lossless:
        cmd = ["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264", "-crf", "0",
               "-preset", "ultrafast", "-pix_fmt", "yuv444p", "-loglevel", "warning", output_path]
    else:
        cmd = ["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264", "-crf", "18",
               "-preset", "fast", "-pix_fmt", "yuv420p", "-loglevel", "warning", output_path]
    subprocess.run(cmd, check=True, capture_output=True)
    os.remove(tmp)


def pad_frames(frames: list, target: int) -> list:
    """Pad frame list to target length by repeating the last frame."""
    if len(frames) >= target:
        return frames
    return frames + [frames[-1]] * (target - len(frames))


def write_clip(rgb_frames, seg_ids_frames, name, output_dir, fps, target_w, target_frames):
    """Resize, pad, colorize, and encode a clip. Returns info dict or None."""
    if not rgb_frames or not seg_ids_frames:
        return None

    # Resize
    resized_rgb = []
    resized_seg = []
    for rgb, seg_ids in zip(rgb_frames, seg_ids_frames):
        src_h, src_w = rgb.shape[:2]
        target_h = int(target_w * src_h / src_w)
        # Round to even for video encoding
        target_h = target_h + (target_h % 2)
        r_rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        r_seg = cv2.resize(seg_ids, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        resized_rgb.append(r_rgb)
        resized_seg.append(r_seg)

    # Pad to target frame count
    orig_count = len(resized_rgb)
    resized_rgb = pad_frames(resized_rgb, target_frames)
    resized_seg = pad_frames(resized_seg, target_frames)

    # Colorize seg
    seg_colored = [colorize_openh(s) for s in resized_seg]

    # Encode
    rgb_path = os.path.join(output_dir, f"{name}.mp4")
    seg_path = os.path.join(output_dir, f"{name}.seg.mp4")
    json_path = os.path.join(output_dir, f"{name}.json")

    encode_video_cv2(resized_rgb, rgb_path, fps=fps, lossless=False)
    encode_video_cv2(seg_colored, seg_path, fps=fps, lossless=True)
    with open(json_path, "w") as f:
        json.dump({"short": DEFAULT_CAPTION}, f)

    return {"video": f"{name}.mp4", "num_frames": len(resized_rgb), "orig_frames": orig_count}


# =============================================================================
# Per-dataset clip generators
# =============================================================================
def collect_atlas120k(data_root, min_frames, splits=("train", "val", "test")):
    """Yield (clip_id, rgb_paths, mask_paths, mask_type) for Atlas120k."""
    for split in splits:
        clip_dirs = sorted(glob.glob(os.path.join(data_root, split, "cholecystectomy/*/clip_*/")))
        for clip_dir in clip_dirs:
            parts = clip_dir.rstrip("/").split("/")
            vid_id, clip_id = parts[-2], parts[-1]
            imgs = sorted(glob.glob(os.path.join(clip_dir, "images/*.jpg")))
            masks = sorted(glob.glob(os.path.join(clip_dir, "masks/*.png")))
            if len(imgs) < min_frames or len(imgs) != len(masks):
                continue
            # Skip RGB-mode masks
            if Image.open(masks[0]).mode == "RGB":
                continue
            name = f"atlas_{vid_id}_{clip_id}"
            yield name, imgs, masks, "atlas"


def collect_cholecseg8k(data_root, min_frames):
    """Yield clips from CholecSeg8k."""
    groups = sorted(glob.glob(os.path.join(data_root, "video*/video*/")))
    for g in groups:
        imgs = sorted(glob.glob(os.path.join(g, "*_endo.png")))
        masks = sorted(glob.glob(os.path.join(g, "*_endo_color_mask.png")))
        if len(imgs) < min_frames or len(imgs) != len(masks):
            continue
        parts = g.rstrip("/").split("/")
        name = f"cs8k_{parts[-2]}_{parts[-1]}"
        yield name, imgs, masks, "cholecseg"


def collect_endoscapes(data_root, min_frames):
    """Yield clips from Endoscapes (grouped by video ID)."""
    seg_dir = os.path.join(data_root, "semseg")
    seg_files = sorted(glob.glob(os.path.join(seg_dir, "*.png")))

    # Group by video ID
    vid_groups = {}
    for sf in seg_files:
        fname = os.path.basename(sf).replace(".png", "")
        vid_id = fname.split("_")[0]
        vid_groups.setdefault(vid_id, []).append(sf)

    for vid_id, seg_list in sorted(vid_groups.items()):
        if len(seg_list) < min_frames:
            continue
        # Find matching images
        img_list = []
        for sf in seg_list:
            fname = os.path.basename(sf).replace(".png", ".jpg")
            for subdir in ("train", "val", "test"):
                candidate = os.path.join(data_root, subdir, fname)
                if os.path.exists(candidate):
                    img_list.append(candidate)
                    break
        if len(img_list) != len(seg_list):
            continue
        name = f"endo_{vid_id}"
        yield name, img_list, seg_list, "endoscapes"


def collect_heisurf(data_root, min_frames):
    """Yield clips from HeiSurf."""
    cases = sorted(glob.glob(os.path.join(data_root, "Segmentation/*/")))
    for c in cases:
        masks = sorted(glob.glob(os.path.join(c, "*.png")))
        if len(masks) < min_frames:
            continue
        case_id = os.path.basename(c.rstrip("/"))
        img_dir = c.replace("/Segmentation/", "/Frames/")
        imgs = [os.path.join(img_dir, os.path.basename(m)) for m in masks]
        if not all(os.path.exists(i) for i in imgs):
            continue
        name = f"heisurf_{case_id}"
        yield name, imgs, masks, "heisurf"


# =============================================================================
# Mask loading dispatch
# =============================================================================
def load_openh_ids(mask_path: str, mask_type: str) -> np.ndarray:
    """Load a mask file and return Open-H class ID array (H, W) uint8."""
    if mask_type == "atlas":
        m = np.array(Image.open(mask_path))
        return ATLAS_REMAP_LUT[m]
    elif mask_type == "cholecseg":
        m = np.array(Image.open(mask_path).convert("RGB"))
        return remap_rgb_mask(m, CHOLECSEG_COLOR_MAP)
    elif mask_type == "endoscapes":
        m = np.array(Image.open(mask_path))
        return ENDOSCAPES_REMAP_LUT[m]
    elif mask_type == "heisurf":
        m = np.array(Image.open(mask_path).convert("RGB"))
        return remap_rgb_mask(m, HEISURF_COLOR_MAP)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def process_clip(name, img_paths, mask_paths, mask_type, output_dir, fps, target_w, target_frames):
    """Process a single clip end-to-end. Returns info dict or None."""
    try:
        rgb_frames = [np.array(Image.open(p).convert("RGB")) for p in img_paths]
        seg_frames = [load_openh_ids(p, mask_type) for p in mask_paths]
        return write_clip(rgb_frames, seg_frames, name, output_dir, fps, target_w, target_frames)
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        return None


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Prepare unified Open-H training data from 4 datasets")
    parser.add_argument("--output-root", required=True, help="Output directory")
    parser.add_argument("--atlas120k", default=None, help="Atlas120k root (contains train/val/test)")
    parser.add_argument("--cholecseg8k", default=None, help="CholecSeg8k root (contains video*/)")
    parser.add_argument("--endoscapes", default=None, help="Endoscapes root (contains semseg/, train/, etc.)")
    parser.add_argument("--heisurf", default=None, help="HeiSurf root (contains Segmentation/, Frames/)")
    parser.add_argument("--min-frames", type=int, default=20, help="Min original frames per clip")
    parser.add_argument("--target-frames", type=int, default=93, help="Pad clips to this length")
    parser.add_argument("--width", type=int, default=1280, help="Output width")
    parser.add_argument("--fps", type=int, default=16, help="Output FPS")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Count clips without writing")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_root, "train")
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Unified Open-H data preparation")
    print(f"  Output:        {args.output_root}")
    print(f"  Width:         {args.width}")
    print(f"  Min frames:    {args.min_frames}")
    print(f"  Target frames: {args.target_frames}")
    print(f"  FPS:           {args.fps}")

    # Collect all clips
    all_clips = []

    if args.atlas120k and os.path.isdir(args.atlas120k):
        clips = list(collect_atlas120k(args.atlas120k, args.min_frames))
        print(f"\n  Atlas120k:    {len(clips)} clips from {args.atlas120k}")
        all_clips.extend(clips)
    else:
        print(f"\n  Atlas120k:    SKIPPED (not found or not specified)")

    if args.cholecseg8k and os.path.isdir(args.cholecseg8k):
        clips = list(collect_cholecseg8k(args.cholecseg8k, args.min_frames))
        print(f"  CholecSeg8k:  {len(clips)} clips from {args.cholecseg8k}")
        all_clips.extend(clips)
    else:
        print(f"  CholecSeg8k:  SKIPPED")

    if args.endoscapes and os.path.isdir(args.endoscapes):
        clips = list(collect_endoscapes(args.endoscapes, args.min_frames))
        print(f"  Endoscapes:   {len(clips)} clips from {args.endoscapes}")
        all_clips.extend(clips)
    else:
        print(f"  Endoscapes:   SKIPPED")

    if args.heisurf and os.path.isdir(args.heisurf):
        clips = list(collect_heisurf(args.heisurf, args.min_frames))
        print(f"  HeiSurf:      {len(clips)} clips from {args.heisurf}")
        all_clips.extend(clips)
    else:
        print(f"  HeiSurf:      SKIPPED")

    total_orig_frames = sum(len(imgs) for _, imgs, _, _ in all_clips)
    print(f"\n  Total clips:  {len(all_clips)}")
    print(f"  Total frames: {total_orig_frames} (before padding)")

    if args.dry_run:
        # Show per-dataset breakdown
        for name, imgs, masks, mtype in all_clips:
            pad = max(0, args.target_frames - len(imgs))
            print(f"    {name}: {len(imgs)} frames (+{pad} pad)")
        print("\n  DRY RUN — no files written")
        return

    # Process clips
    results = []
    if args.workers <= 1:
        for i, (name, imgs, masks, mtype) in enumerate(all_clips):
            r = process_clip(name, imgs, masks, mtype, output_dir, args.fps, args.width, args.target_frames)
            if r:
                results.append(r)
                print(f"  [{len(results)}/{len(all_clips)}] {r['video']} "
                      f"({r['orig_frames']}→{r['num_frames']} frames)")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(process_clip, name, imgs, masks, mtype,
                          output_dir, args.fps, args.width, args.target_frames): name
                for name, imgs, masks, mtype in all_clips
            }
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)
                    print(f"  [{len(results)}/{len(all_clips)}] {r['video']} "
                          f"({r['orig_frames']}→{r['num_frames']} frames)")

    # Sort and write index
    results.sort(key=lambda r: r["video"])
    index = {"training": [{"video": r["video"]} for r in results]}
    index_path = os.path.join(args.output_root, "train.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    total_output_frames = sum(r["num_frames"] for r in results)
    total_orig = sum(r["orig_frames"] for r in results)
    print(f"\n{'='*60}")
    print(f"Done: {len(results)} videos, {total_orig} original frames → {total_output_frames} output frames")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
