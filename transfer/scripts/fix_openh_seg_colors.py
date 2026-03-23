"""Re-encode Open-H segmentation videos with exact palette colors.

The Open-H seg maps use the correct palette but video compression shifts colors
by 30-55 units. This script snaps each pixel to the nearest Open-H class color
and re-encodes as lossless H.264.

Usage:
    cd transfer
    python scripts/fix_openh_seg_colors.py inputs/open_h/nonexpert-2026-01-15_16-50-48/segmentation/
    python scripts/fix_openh_seg_colors.py inputs/open_h/nonexpert-2026-01-15_16-50-48/segmentation/episode_000043.mp4
"""

import glob
import os
import subprocess
import sys

import cv2
import numpy as np
from decord import VideoReader, cpu

OPENH_PALETTE = np.array([
    [0, 0, 0],        # 0: Background
    [64, 64, 64],     # 1: Abdominal Wall
    [255, 73, 40],    # 2: Liver
    [255, 45, 255],   # 3: Gallbladder
    [57, 4, 105],     # 4: Fat
    [137, 126, 8],    # 5: Connective Tissue
    [100, 228, 3],    # 6: Instruments
    [74, 74, 138],    # 7: Other Anatomy
], dtype=np.float32)


def snap_frame_to_palette(frame_rgb: np.ndarray) -> np.ndarray:
    """Snap each pixel to the nearest Open-H palette color."""
    h, w, _ = frame_rgb.shape
    flat = frame_rgb.reshape(-1, 3).astype(np.float32)

    # Compute distance to each palette color: (N, 8)
    dists = np.linalg.norm(flat[:, None, :] - OPENH_PALETTE[None, :, :], axis=2)
    nearest = dists.argmin(axis=1)

    out = OPENH_PALETTE[nearest].astype(np.uint8)
    return out.reshape(h, w, 3)


def fix_video(input_path: str, output_path: str):
    """Read seg video, snap colors, re-encode lossless."""
    vr = VideoReader(input_path, ctx=cpu(0))
    num_frames = len(vr)
    fps = int(round(vr.get_avg_fps()))

    print(f"  {os.path.basename(input_path)}: {num_frames} frames @ {fps} fps")

    # Process all frames
    fixed_frames = []
    for i in range(num_frames):
        frame = vr[i].asnumpy()
        fixed = snap_frame_to_palette(frame)
        fixed_frames.append(fixed)

    # Encode: write PNG frames to temp dir, then ffmpeg to lossless video
    h, w = fixed_frames[0].shape[:2]
    tmp_dir = output_path + ".frames"
    os.makedirs(tmp_dir, exist_ok=True)
    for i, f in enumerate(fixed_frames):
        cv2.imwrite(os.path.join(tmp_dir, f"frame_{i:06d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    tmp_out = output_path + ".fixed.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmp_dir, "frame_%06d.png"),
        "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
        "-pix_fmt", "yuv444p", "-loglevel", "warning",
        tmp_out,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Clean up temp frames and replace original
    import shutil
    shutil.rmtree(tmp_dir)
    os.replace(tmp_out, output_path)

    # Verify
    vr2 = VideoReader(output_path, ctx=cpu(0))
    test_frame = vr2[num_frames // 2].asnumpy()
    unique_colors = np.unique(test_frame.reshape(-1, 3), axis=0)
    print(f"    -> {len(unique_colors)} unique colors (should be <= 8)")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <seg_video_or_dir>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        videos = sorted(glob.glob(os.path.join(target, "*.mp4")))
    else:
        videos = [target]

    print(f"Fixing {len(videos)} seg videos to exact Open-H palette")
    for v in videos:
        fix_video(v, v)  # overwrite in place

    print("Done.")


if __name__ == "__main__":
    main()
