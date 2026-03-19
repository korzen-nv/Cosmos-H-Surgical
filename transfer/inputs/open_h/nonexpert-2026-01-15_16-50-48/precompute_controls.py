#!/usr/bin/env python3
"""Precompute edge and vis (blur) control maps for episode 043.

Generates:
  edge/episode_000043.mp4  — Canny edge map (medium preset: t_lower=100, t_upper=200)
  vis/episode_000043.mp4   — Blurred vis map (medium preset: bilateral d=30 + downup factor 10)

Uses the same processing as the inference pipeline's on-the-fly generation,
so precomputed maps are identical to what the model would see.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F


def generate_edge_map(in_path: str, out_path: str, t_lower: int = 100, t_upper: int = 200) -> None:
    """Generate Canny edge map matching AddControlInputEdge(preset_strength='medium')."""
    cap = cv2.VideoCapture(in_path)
    assert cap.isOpened(), f"Could not open: {in_path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        edges = cv2.Canny(frame, t_lower, t_upper)
        out.write(edges)

    cap.release()
    out.release()
    print(f"Edge map → {out_path}")


def generate_vis_map(
    in_path: str,
    out_path: str,
    blur_downsize_factor: int = 2,
    bilateral_d: int = 30,
    bilateral_sigma_color: int = 150,
    bilateral_sigma_space: int = 100,
    downup_factor: int = 10,
) -> None:
    """Generate blurred vis map matching AddControlInputBlur(preset='medium').

    Pipeline: downscale by blur_downsize_factor → bilateral filter → upscale → downsize by downup_factor → upsize.
    """
    cap = cv2.VideoCapture(in_path)
    assert cap.isOpened(), f"Could not open: {in_path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    # frames as CTHW numpy (matching augmentor's expected format)
    frames_np = np.stack(frames).transpose((3, 0, 1, 2))  # (T,H,W,C) → (C,T,H,W)

    # Step 1: Downscale before blur
    small_frames = []
    for img in frames_np.transpose((1, 2, 3, 0)):  # → (T,H,W,C)
        small = cv2.resize(img, (W // blur_downsize_factor, H // blur_downsize_factor), interpolation=cv2.INTER_AREA)
        small_frames.append(small)
    frames_np = np.stack(small_frames).transpose((3, 0, 1, 2))  # → (C,T,H,W)

    # Step 2: Bilateral filter
    blurred = []
    for img in frames_np.transpose((1, 2, 3, 0)):  # → (T,H,W,C)
        img = cv2.bilateralFilter(img, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
        blurred.append(img)
    frames_np = np.stack(blurred).transpose((3, 0, 1, 2))

    # Step 3: Upscale back to original size
    upscaled = []
    for img in frames_np.transpose((1, 2, 3, 0)):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        upscaled.append(img)
    frames_np = np.stack(upscaled).transpose((3, 0, 1, 2))

    # Step 4: Downsize/upsize (bicubic) by downup_factor — matching torchvision transforms
    tensor = torch.from_numpy(frames_np)  # (C,T,H,W)
    tensor = transforms_F.resize(
        tensor,
        size=(H // downup_factor, W // downup_factor),
        interpolation=transforms_F.InterpolationMode.BICUBIC,
        antialias=True,
    )
    tensor = transforms_F.resize(
        tensor,
        size=(H, W),
        interpolation=transforms_F.InterpolationMode.BICUBIC,
        antialias=True,
    )
    frames_np = tensor.numpy()  # (C,T,H,W)

    # Write output video
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H), isColor=True)
    for img in frames_np.transpose((1, 2, 3, 0)):  # → (T,H,W,C)
        img = np.clip(img, 0, 255).astype(np.uint8)
        out.write(img)
    out.release()
    print(f"Vis map → {out_path}")


def main():
    base = Path(__file__).parent
    color_path = str(base / "color" / "episode_000043.mp4")

    generate_edge_map(color_path, str(base / "edge" / "episode_000043.mp4"))
    generate_vis_map(color_path, str(base / "vis" / "episode_000043.mp4"))


if __name__ == "__main__":
    main()
