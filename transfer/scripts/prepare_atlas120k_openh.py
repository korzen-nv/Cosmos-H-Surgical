"""Prepare atlas120k cholecystectomy data for Cosmos-Transfer2 segmentation training.

Converts atlas120k frame sequences + indexed masks into the format expected by
SingleViewTransferDatasetJSON:
    <name>.mp4       — RGB video at 1280x720
    <name>.seg.mp4   — Color-coded open-h segmentation video at 1280x720
    <name>.json      — Caption file {"short": "..."}
    train.json       — Index file {"training": [{"video": "..."}]}

Usage:
    cd transfer
    python scripts/prepare_atlas120k_openh.py
    python scripts/prepare_atlas120k_openh.py --splits train,val
    python scripts/prepare_atlas120k_openh.py --min-frames 93 --fps 16
    python scripts/prepare_atlas120k_openh.py --dry-run  # preview without writing
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

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
    # Additional structures (may appear in other surgical types)
    43: 7,  # Mesocolon -> Other Anatomy
    44: 7,  # Adrenal Gland -> Other Anatomy
    45: 7,  # Pancreas -> Other Anatomy
    46: 7,  # Duodenum -> Other Anatomy
}

# Build a fast lookup table (covers class IDs 0-255)
_REMAP_LUT = np.zeros(256, dtype=np.uint8)
for atlas_id, openh_id in ATLAS_TO_OPENH.items():
    _REMAP_LUT[atlas_id] = openh_id

# Build colorization LUT: openh class ID -> RGB
_COLOR_LUT = np.zeros((256, 3), dtype=np.uint8)
for cls_id, (_, color) in OPENH_CLASSES.items():
    _COLOR_LUT[cls_id] = color

DEFAULT_CAPTION = (
    "A laparoscopic cholecystectomy surgical video showing the "
    "dissection and removal of the gallbladder from the liver bed. "
    "Surgical instruments manipulate tissue in the hepatocystic triangle "
    "region, with liver, gallbladder, fat, and connective tissue visible."
)


def remap_and_colorize(mask_arr: np.ndarray) -> np.ndarray:
    """Remap atlas class IDs to open-h and colorize. Input: (H, W) uint8. Output: (H, W, 3) uint8."""
    openh_ids = _REMAP_LUT[mask_arr]
    return _COLOR_LUT[openh_ids]


def encode_video_cv2(frames: list[np.ndarray], output_path: str, fps: int, lossless: bool = False):
    """Encode frames to MP4 using cv2.VideoWriter."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()

    if lossless:
        # Re-encode with ffmpeg for lossless H.264 (decord reads H.264 better than mp4v)
        tmp_path = output_path + ".tmp.mp4"
        os.rename(output_path, tmp_path)
        cmd = [
            "ffmpeg", "-y", "-i", tmp_path,
            "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
            "-pix_fmt", "yuv444p", "-loglevel", "warning",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(tmp_path)
    else:
        # Re-encode with ffmpeg for H.264 compatibility (decord prefers H.264 over mp4v)
        tmp_path = output_path + ".tmp.mp4"
        os.rename(output_path, tmp_path)
        cmd = [
            "ffmpeg", "-y", "-i", tmp_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-loglevel", "warning",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(tmp_path)


def process_clip(
    clip_dir: str,
    output_dir: str,
    video_id: str,
    clip_id: str,
    fps: int,
    target_w: int,
    target_h: int,
    dry_run: bool = False,
) -> dict | None:
    """Process a single atlas120k clip into training format.

    Returns clip info dict on success, None on skip.
    """
    images_dir = os.path.join(clip_dir, "images")
    masks_dir = os.path.join(clip_dir, "masks")

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

    if not image_files or not mask_files:
        return None

    # Verify mask mode (skip RGB-mode masks)
    test_mask = Image.open(mask_files[0])
    if test_mask.mode == "RGB":
        return None

    # Verify frame counts match
    if len(image_files) != len(mask_files):
        print(f"  WARN: frame count mismatch in {video_id}/{clip_id}: "
              f"{len(image_files)} images vs {len(mask_files)} masks, skipping")
        return None

    num_frames = len(image_files)
    name = f"{video_id}_{clip_id}"

    if dry_run:
        return {"video": f"{name}.mp4", "num_frames": num_frames}

    # Read, resize, and collect frames
    rgb_frames = []
    seg_frames = []

    for img_path, mask_path in zip(image_files, mask_files):
        # RGB frame
        rgb = cv2.imread(img_path)
        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb_frames.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

        # Mask frame: remap + colorize
        mask = np.array(Image.open(mask_path))
        # Resize mask BEFORE remapping to preserve exact class IDs (INTER_NEAREST)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        seg_rgb = remap_and_colorize(mask)
        seg_frames.append(seg_rgb)

    # Encode RGB video
    rgb_path = os.path.join(output_dir, f"{name}.mp4")
    encode_video_cv2(rgb_frames, rgb_path, fps=fps, lossless=False)

    # Encode seg video (lossless to preserve class colors)
    seg_path = os.path.join(output_dir, f"{name}.seg.mp4")
    encode_video_cv2(seg_frames, seg_path, fps=fps, lossless=True)

    # Write caption
    json_path = os.path.join(output_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump({"short": DEFAULT_CAPTION}, f)

    return {"video": f"{name}.mp4", "num_frames": num_frames}


def process_split(
    data_root: str,
    output_root: str,
    split: str,
    fps: int,
    target_w: int,
    target_h: int,
    min_frames: int,
    max_workers: int,
    dry_run: bool,
) -> None:
    """Process all cholecystectomy clips in a split."""
    split_dir = os.path.join(data_root, split, "cholecystectomy")
    output_dir = os.path.join(output_root, split)

    if not os.path.isdir(split_dir):
        print(f"Split directory not found: {split_dir}, skipping")
        return

    # Collect clips
    clip_dirs = sorted(glob.glob(os.path.join(split_dir, "*/clip_*/")))
    print(f"\n{'='*60}")
    print(f"Split: {split} — {len(clip_dirs)} clips found")

    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    # Process clips
    results = []
    tasks = []
    for clip_dir in clip_dirs:
        parts = clip_dir.rstrip("/").split("/")
        video_id = parts[-2]
        clip_id = parts[-1]

        # Quick frame count check
        n_images = len(glob.glob(os.path.join(clip_dir, "images", "*.jpg")))
        if n_images < min_frames:
            continue

        tasks.append((clip_dir, output_dir, video_id, clip_id, fps, target_w, target_h, dry_run))

    print(f"Processing {len(tasks)} clips (>= {min_frames} frames, P-mode masks)")

    if max_workers <= 1 or dry_run:
        for i, args in enumerate(tasks):
            result = process_clip(*args)
            if result:
                results.append(result)
                if not dry_run:
                    print(f"  [{len(results)}/{len(tasks)}] {result['video']} ({result['num_frames']} frames)")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_clip, *args): args for args in tasks}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                    if not dry_run:
                        print(f"  [{len(results)}/{len(tasks)}] {result['video']} ({result['num_frames']} frames)")

    # Sort results for deterministic ordering
    results.sort(key=lambda r: r["video"])

    # Write index JSON
    if not dry_run and results:
        index_path = os.path.join(output_root, f"{split}.json")
        index = {"training": [{"video": r["video"]} for r in results]}
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Index written: {index_path} ({len(results)} videos)")

    # Summary
    total_frames = sum(r["num_frames"] for r in results)
    print(f"\nSplit '{split}' summary:")
    print(f"  Videos:       {len(results)}")
    print(f"  Total frames: {total_frames}")
    if results:
        frame_counts = [r["num_frames"] for r in results]
        print(f"  Frame range:  {min(frame_counts)} - {max(frame_counts)}")
    if dry_run:
        print("  (DRY RUN — no files written)")


def main():
    parser = argparse.ArgumentParser(description="Prepare atlas120k for Cosmos-Transfer2 seg training")
    parser.add_argument("--data-root", default="../data/atlas120k", help="Path to atlas120k dataset")
    parser.add_argument("--output-root", default="../data/atlas120k_openh", help="Output directory")
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits to process")
    parser.add_argument("--min-frames", type=int, default=93, help="Minimum frames per clip")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--width", type=int, default=1280, help="Output width")
    parser.add_argument("--height", type=int, default=720, help="Output height")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    args = parser.parse_args()

    print(f"Atlas120k -> Open-H data preparation")
    print(f"  Input:    {args.data_root}")
    print(f"  Output:   {args.output_root}")
    print(f"  Target:   {args.width}x{args.height} @ {args.fps} fps")
    print(f"  Min frames: {args.min_frames}")

    for split in args.splits.split(","):
        process_split(
            data_root=args.data_root,
            output_root=args.output_root,
            split=split.strip(),
            fps=args.fps,
            target_w=args.width,
            target_h=args.height,
            min_frames=args.min_frames,
            max_workers=args.workers,
            dry_run=args.dry_run,
        )

    print(f"\nDone.")


if __name__ == "__main__":
    main()
