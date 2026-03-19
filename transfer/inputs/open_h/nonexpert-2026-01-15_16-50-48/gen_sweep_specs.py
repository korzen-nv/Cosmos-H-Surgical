#!/usr/bin/env python3
"""Generate 256 multicontrol weight sweep specs (4^4 full factorial)."""

import itertools
import json
from pathlib import Path

WEIGHTS = [0.25, 0.5, 0.75, 1.0]
CONTROLS = ["depth", "edge", "vis", "seg"]

BASE_DIR = "inputs/open_h/nonexpert-2026-01-15_16-50-48"

TEMPLATE = {
    "prompt": "real surgery scene: robotic cholecystectomy procedure with surgical instruments manipulating tissue",
    "negative_prompt": (
        "The video captures a game playing, with bad crappy graphics and cartoonish frames. "
        "It represents a recording of old outdated games. The lighting looks very fake. "
        "The textures are very raw and basic. The geometries are very primitive. "
        "The images are very pixelated and of poor CG quality. "
        "There are many subtitles in the footage. Overall, the video is unrealistic at all."
    ),
    "video_path": f"{BASE_DIR}/color/episode_000043.mp4",
    "guidance": 3,
    "sigma_max": "70",
    "max_frames": 1,
    "num_video_frames_per_chunk": 1,
}

CONTROL_PATHS = {
    "depth": f"{BASE_DIR}/depth/episode_000043.mp4",
    "edge": f"{BASE_DIR}/edge/episode_000043.mp4",
    "vis": f"{BASE_DIR}/vis/episode_000043.mp4",
    "seg": f"{BASE_DIR}/segmentation/episode_000043.mp4",
}


def weight_label(w: float) -> str:
    return f"{int(w * 100):03d}"


def main():
    out_path = Path(__file__).parent / "ep043_sweep_all.jsonl"
    specs = []

    for d_w, e_w, v_w, s_w in itertools.product(WEIGHTS, repeat=4):
        name = f"ep043_d{weight_label(d_w)}_e{weight_label(e_w)}_v{weight_label(v_w)}_s{weight_label(s_w)}"
        spec = {
            "name": name,
            **TEMPLATE,
            "depth": {"control_weight": d_w, "control_path": CONTROL_PATHS["depth"]},
            "edge": {"control_weight": e_w, "control_path": CONTROL_PATHS["edge"]},
            "vis": {"control_weight": v_w, "control_path": CONTROL_PATHS["vis"]},
            "seg": {"control_weight": s_w, "control_path": CONTROL_PATHS["seg"]},
        }
        specs.append(spec)

    with open(out_path, "w") as f:
        for spec in specs:
            f.write(json.dumps(spec) + "\n")

    print(f"Generated {len(specs)} specs → {out_path}")
    assert len(specs) == 256, f"Expected 256, got {len(specs)}"

    # Verify all names are unique
    names = [s["name"] for s in specs]
    assert len(set(names)) == 256, "Duplicate names found!"


if __name__ == "__main__":
    main()
