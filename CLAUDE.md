# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cosmos-H-Surgical is a surgical video world foundation model suite based on NVIDIA Cosmos. It contains two independent sub-projects with separate environments:

- **predict/** — Cosmos-H-Surgical-Predict: Image-to-video generation (Image2World, Video2World, Text2World) using a 2B diffusion model fine-tuned from Cosmos-Predict2.5
- **transfer/** — Cosmos-H-Surgical-Transfer: Control-conditioned video generation (depth, edge, segmentation, blur) using a 2B model fine-tuned from Cosmos-Transfer2.5

Each sub-project is a self-contained Python package with its own `pyproject.toml`, virtual environment, and Docker setup.

## Environment Setup

Each sub-project uses **uv** as the package manager with a workspace layout (`packages/` contains `cosmos-oss`, `cosmos-cuda`, `cosmos-gradio`). They must be set up independently:

```bash
cd predict  # or cd transfer
uv python install
uv sync --extra=cu128  # or --extra=cu130 for Blackwell/Jetson
source .venv/bin/activate
```

Requires: NVIDIA Ampere+ GPU, Linux x86-64, CUDA 12.8+, Python 3.10+. Model weights auto-download from HuggingFace (requires `HF_TOKEN`).

## Common Commands

### Inference
```bash
# Predict (Image2World)
cd predict
python examples/inference.py -i assets/base/coagulation.json -o outputs/base_video2world --inference-type=video2world

# Transfer (control-conditioned)
cd transfer
python examples/inference.py -i assets/coagulation_example/depth/coagulation_depth_spec.json -o outputs/depth
```

### Linting
Both sub-projects use **ruff** (line-length=120, target Python 3.10):
```bash
cd predict  # or transfer
ruff check .
ruff format .
```

### Testing
Test files use the `*_test.py` naming convention (not `test_*.py`). Pytest is configured per sub-project:
```bash
cd predict  # or transfer
pytest                           # run all tests (ignores _src/ and packages/)
pytest tests/assets_test.py      # run a single test file
pytest --num-gpus=1              # run GPU tests requiring 1 GPU
pytest --levels=0                # run only level-0 tests
pytest --manual                  # include manual-only tests
```

Key pytest markers: `@pytest.mark.manual`, `@pytest.mark.level(n)` (0-2), `@pytest.mark.gpus(n)`. GPU tests run in subprocesses (`CUDA_VISIBLE_DEVICES=""` in main process).

### Docker
```bash
cd predict  # or transfer
image_tag=$(docker build -f Dockerfile -q .)
docker run -it --runtime=nvidia --ipc=host --rm -v .:/workspace -v /workspace/.venv -v /root/.cache:/root/.cache -e HF_TOKEN="$HF_TOKEN" $image_tag
```

## Architecture

### Package Structure
Both sub-projects share the same internal layout:
- `cosmos_{predict2,transfer2}/` — Top-level package with config, inference entry points, and experiment definitions
- `cosmos_{predict2,transfer2}/_src/` — Internal implementation (excluded from tests by default):
  - `imaginaire/` — Core ML framework: attention backends (Flash2/3, NATTEN), datasets/dataloaders, checkpointing, callbacks, guardrails
  - `predict2/` — Predict model implementation (diffusion configs, rectified flow)
  - `transfer2/` — Transfer model implementation (control-conditioned generation)
- `packages/cosmos-oss/` — Shared dependency package (`cosmos-oss`) with common utils, checkpoint management, HuggingFace integration
- `packages/cosmos-cuda/` — CUDA sentinel package for GPU dependency resolution
- `packages/cosmos-gradio/` — Gradio UI components

### Config & CLI
Both sub-projects use **pydantic** models for configuration and **tyro** for CLI argument parsing. Inference parameters are loaded from JSON/JSONL/YAML files via `-i` flag, with CLI overrides taking precedence.

### Key Differences Between Predict and Transfer
- **Predict** supports three inference types: `text2world`, `image2world`, `video2world`
- **Transfer** uses control subcommands: `edge`, `depth`, `vis` (blur), `seg` (segmentation) and accepts control maps as additional input

## Licensing
Source code is Apache 2.0. Model weights use NVIDIA-OneWay-Noncommercial-License.
