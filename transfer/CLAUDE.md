# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cosmos-H-Surgical-Transfer is a control-conditioned surgical video generation model (Multi-ControlNet) fine-tuned from Cosmos-Transfer2.5-2B. It generates realistic surgical videos from control maps: **depth**, **edge**, **segmentation**, and **blur** (vis). Part of the NVIDIA Cosmos-H-Surgical suite alongside the sibling `predict/` sub-project.

## Environment Setup

Requires: NVIDIA Ampere+ GPU, Linux x86-64, CUDA 12.8+, Python 3.10+, `HF_TOKEN` for model weight download.

```bash
uv python install
uv sync --extra=cu128       # or --extra=cu130 for Blackwell/Jetson
source .venv/bin/activate
```

The `just` task runner wraps most commands. `just install` handles CUDA detection and syncing.

## Common Commands

### Inference
```bash
# Single control (depth)
python examples/inference.py -i assets/coagulation_example/depth/coagulation_depth_spec.json -o outputs/depth depth

# Other controls: edge, vis (blur), seg (segmentation) — passed as subcommand
python examples/inference.py -i <spec.json> -o <output_dir> edge

# Multi-control (auto-detected from spec files with multiple control types)
python examples/inference.py -i assets/multicontrol.jsonl -o outputs/multicontrol depth
```

### Linting & Type Checking
```bash
ruff check .                # lint (line-length=120, target py310)
ruff format .               # format
just lint                   # full pre-commit (includes ruff + more)
just pyrefly                # static type checking (pyrefly)
```

Ruff rules: E, F, I (isort), TID252 (no relative imports), T10 (no debugger). Ignored: E402, E501, E721, E741, F541, F811, F841.

### Testing
Test files use `*_test.py` naming (not `test_*.py`). Pytest ignores `_src/`, `packages/`, `imaginaire/`, `projects/` by default.

```bash
pytest                                    # all tests (CPU only, no GPU in main process)
pytest tests/assets_test.py               # single test file
just test-single tests/assets_test.py     # single test (with --manual --capture=no)
just test-cpu                             # CPU tests (level-0, parallel)
just test-gpu                             # GPU tests (1-GPU and MAX_GPUS, level-0)
just test                                 # pyrefly + CPU + GPU
pytest --num-gpus=1                       # tests requiring exactly 1 GPU
pytest --num-gpus=0                       # CPU-only tests
pytest --levels=0                         # level-0 tests only
pytest --levels=0,1                       # levels 0 and 1
pytest --manual                           # include manual-only tests
```

**Markers:** `@pytest.mark.manual`, `@pytest.mark.level(n)` (0=smoke, 1=partial E2E, 2=full E2E), `@pytest.mark.gpus(n)` (0, 1, or MAX_GPUS). Level-0 tests allow only 0 or 1 GPU. Level-1/2 tests allow 0, 1, or MAX_GPUS.

**GPU isolation:** `CUDA_VISIBLE_DEVICES=""` is set in the main pytest process. GPU tests get devices allocated per xdist worker. Each worker gets a unique `MASTER_PORT` (12341 + worker_index).

### Docker
```bash
just docker-cu128           # build and run CUDA 12.8 container
just docker-cu130           # build and run CUDA 13.0 container
```

## Architecture

### Config & CLI System
- **Pydantic models** define all configuration (`cosmos_transfer2/config.py`): `SetupArguments`, `InferenceArguments`, control-specific configs (`EdgeConfig`, `DepthConfig`, `BlurConfig`, `SegConfig`)
- **tyro** CLI with control type as a subcommand: `edge`, `depth`, `vis`, `seg`
- Inference parameters are loaded from JSON/JSONL/YAML via `-i` flag; CLI overrides take precedence
- `InferenceOverrides` pattern: every `InferenceArguments` field is wrapped as `Optional[T]` for selective CLI override

### Inference Pipeline
- `examples/inference.py` → `Control2WorldInference` (`cosmos_transfer2/inference.py`) → `ControlVideo2WorldInference` (`_src/transfer2/inference/inference_pipeline.py`)
- Single-control uses the matching variant checkpoint; multi-control (multiple control types in batch) loads ALL four control checkpoints and uses the `multibranch` experiment
- Model checkpoints are registered in `MODEL_CHECKPOINTS` dict keyed by `ModelKey(variant, distilled)`
- Experiment configs live in `_src/transfer2/configs/vid2vid_transfer/experiment/`

### Internal Implementation (`_src/`)
- `imaginaire/` — Core ML framework shared across Cosmos models: attention backends (Flash2/3, NATTEN), datasets, checkpointing, distributed training, guardrails, callbacks
- `transfer2/` — Transfer-specific: inference pipeline, control-conditioned diffusion (rectified flow), model networks, configs
- `predict2/`, `predict2_multiview/`, `transfer2_multiview/`, `interactive/`, `reason1/` — Other model variants sharing the imaginaire framework

### Workspace Packages (`packages/`)
- `cosmos-oss` — Shared dependency: checkpoint management, HuggingFace integration, environment init, test fixtures, VQA utilities. Heavy dependency list (PyTorch, transformers, megatron-core, etc.)
- `cosmos-cuda` — CUDA sentinel for GPU dependency resolution. Package init validates that a CUDA extra (`cu128`/`cu130`) was installed
- `cosmos-gradio` — Gradio web UI components

For inference performance benchmarks and optimization ideas, see [`PERFORMANCE.md`](PERFORMANCE.md).

For dataset details, class mappings, and preparation scripts, see [`DATASETS.md`](DATASETS.md) and [`MAPPINGS.md`](MAPPINGS.md).

For cluster access and Slurm usage, see [`../SLURM.md`](../SLURM.md).

## Segmentation ControlNet Training

### Open-H Unified Format
The seg controlnet is trained on 4 cholecystectomy datasets mapped to a unified 8-class Open-H
palette (Background, Abdominal Wall, Liver, Gallbladder, Fat, Connective Tissue, Instruments,
Other Anatomy). See [`MAPPINGS.md`](MAPPINGS.md) for per-dataset class mappings.

**Datasets:** Atlas120k (95 clips), CholecSeg8k (101 clips), Endoscapes (7 clips), HeiSurf (8 clips) — 211 total, ~34k frames. See [`DATASETS.md`](DATASETS.md).

### Data Preparation
```bash
# Prepare all 4 datasets for cluster training (unified Open-H format)
sbatch prepare_unified_data_mars.slurm

# Or Atlas120k only (local)
python scripts/prepare_atlas120k_openh.py --data-root ../data/atlas120k --output-root data/atlas120k_openh
```

### Training on MARS Cluster
```bash
SSH_HOST=pkorzeniowsk@pkorzeniowsk-oci-iad-cs.park.nvidia.com
COSMOS_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk/cosmos/Cosmos-H-Surgical/transfer

# Submit training (1 DGX node, 8x A100, 4h, saves every 100 iters)
ssh $SSH_HOST "cd $COSMOS_ROOT && sbatch run_train_seg_mars.slurm 5000 0.0000432 100"

# Chain continuation job (resumes optimizer state)
ssh $SSH_HOST "cd $COSMOS_ROOT && \
  JOB1=\$(sbatch --parsable run_train_seg_mars.slurm 5000 0.0000432 100) && \
  sbatch --dependency=afterany:\$JOB1 run_train_seg_mars.slurm 10000 0.0000432 100 . . . True"
```

**Args:** `run_train_seg_mars.slurm [MAX_ITER] [LR] [SAVE_ITER] [. . . RESUME_STATE]`
- Default LR: 0.0000432 (2^-14.5)
- RESUME_STATE: `False` for fresh start (loads HF checkpoint), `True` for continuation
- Training speed: ~57 s/step on 8x A100, ~240 iters per 4h job
- Checkpoints: DCP format on lustre, convert to `.pt` for inference via `scripts/convert_distcp_to_pt.py`

### Inference with Fine-tuned Checkpoint
```bash
# Run sweep with fine-tuned seg model (converts DCP→.pt, then 8 parallel inferences)
ssh $SSH_HOST "cd $COSMOS_ROOT && sbatch run_sweep_seg_openh_mars.slurm iter_000000600"

# Local inference with custom checkpoint
python examples/inference.py -i spec.json -o outputs/ --checkpoint-path /path/to/model_ema_bf16.pt seg
```

## MARS Cluster Inference

### Quick Start
```bash
SSH_HOST=pkorzeniowsk@pkorzeniowsk-oci-iad-cs.park.nvidia.com
COSMOS_ROOT=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk/cosmos/Cosmos-H-Surgical/transfer

# Submit inference job (multicontrol example, 8 GPUs)
ssh $SSH_HOST "cd $COSMOS_ROOT && sbatch run_inference_mars.slurm assets/multicontrol.jsonl outputs/mars_test depth 8"

# Monitor
ssh $SSH_HOST "squeue -A healthcareeng_holoscan"
ssh $SSH_HOST "tail -f $COSMOS_ROOT/logs/cosmos_transfer_inf_*.log"

# Download results
scp "$SSH_HOST:$COSMOS_ROOT/outputs/mars_test/*.mp4" .
```

### Slurm Script: `run_inference_mars.slurm`
```bash
sbatch run_inference_mars.slurm [INPUT_FILE] [OUTPUT_DIR] [CONTROL] [NUM_GPUS]
# Defaults: assets/multicontrol.jsonl, outputs/mars_test, depth, 8
```

**Container**: `im4-onelogger.sqsh` (PyTorch 2.5, CUDA 12.6, Python 3.10, 8x A100-80GB)

**Key constraints** (container compatibility):
- `numpy<2` — container's PyTorch 2.5 compiled against NumPy 1.x
- `megatron-core<0.14` — newer versions need NumPy 2
- `deepspeed>=0.16` installed `--no-deps` — fixes pydantic-core 2.33 compat
- Torch-dependent packages (`megatron-core`, `diffusers`, `peft`, `timm`, `sam2`, `transformers`) installed `--no-deps` to prevent upgrading container's PyTorch

**First run** downloads ~20GB of checkpoints (edge/depth/seg/vis + tokenizer + text encoder) to HF cache on Lustre.

### GPU parallelism strategy
Always run **independent single-GPU processes** (one video per GPU), not `torchrun` context parallelism across GPUs. The pipeline uses context parallelism (CP) by default — all GPUs shard the same video temporally — but CP scales sublinearly. Single-GPU peak VRAM is ~53 GB, which fits in A100-80GB with headroom.

```bash
# 8 videos on 8 GPUs — launch 8 independent processes
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$gpu python examples/inference.py \
    -i "input_${gpu}.json" -o "outputs/video_${gpu}" depth &
done
wait
```

This is ~2× faster than `torchrun --nproc_per_node=8` running videos sequentially. Split input specs into per-GPU files (one JSON/JSONL each). No `torchrun` needed — plain `python` with `CUDA_VISIBLE_DEVICES`.

### Cluster Paths
```
/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk/cosmos/Cosmos-H-Surgical/transfer/
├── run_inference_mars.slurm      # Main Slurm inference script
├── build_container.slurm         # NGC image import (if needed)
├── outputs/mars_test/            # Inference results
└── logs/                         # Job logs
```

## Datasets

For full dataset documentation (formats, class encodings, cluster paths), see [`DATASETS.md`](DATASETS.md).

### Accessing datasets
Always use the `/data` symlink (project root `data/` → `/home/pkorzeniowsk/Datasets/`) when referencing dataset paths in scripts and spec files. This keeps paths within the sandbox write allowlist and avoids permission prompts.

Example: `/home/pkorzeniowsk/Projects/cosmos/cosmos-h-surgical/data/Open-H/...`

### Open-H Non-Expert dataset structure
Path: `/data/Open-H/Non-Experts/<session>/videos/chunk-000/observation.images.color/`

The number of clips varies per session, but the **interleaving pattern is fixed**: clips are organized as time sequences of 18 episodes each, with lighting conditions cycling in groups of 6 within every sequence:

| Within-sequence offset | Lighting | Notes |
|---|---|---|
| +0 – +5 | First (dim) | |
| +6 – +11 | Second (bright) | **Best quality — prefer these** |
| +12 – +17 | Third (dim) | |

To find best-lighting clips in any session: episodes 6–11, 24–29, 42–47, 60–65, … (i.e. `18k+6` – `18k+11` for k=0,1,2,…).

**Example session** `nonexpert-2026-01-15_16-50-48`: 72 clips (4 time sequences, episodes 000000–000071).

### Input folder structure
Preprocessed (720p) inputs live under `inputs/` mirroring the dataset hierarchy:

```
inputs/open_h/<session>/
├── color/          # 720p color videos
├── depth/          # 720p depth videos
├── segmentation/   # 720p segmentation videos
├── prompt.txt      # text prompt for this session/run
└── ep<NNN>_<ctrl>.json[l]   # inference spec files
```

```bash
# Downscale color + depth + segmentation for a single episode
CHUNK=/data/Open-H/Non-Experts/<session>/videos/chunk-000
OUT=inputs/open_h/<session>
for mod in color depth segmentation; do
  ffmpeg -y -i $CHUNK/observation.images.$mod/episode_000006.mp4 \
    -vf scale=1280:720 -c:v libx264 -crf 18 -preset fast \
    $OUT/$mod/episode_000006.mp4 -loglevel warning
done
```

### Output folder structure
Outputs use a sequential run-numbered folder under `outputs/open_h/` so that results from different experiments are never overwritten:

```
outputs/open_h/<NN_YYYY_MM_DD>/<session>/
```

- `NN` is a zero-padded two-digit run counter, incrementing per experiment day (`00`, `01`, `02`, …)
- `YYYY_MM_DD` is the date the run was launched
- `<session>` is the dataset session name (e.g. `nonexpert-2026-01-15_16-50-48`)

**Example:** `outputs/open_h/00_2026_03_18/nonexpert-2026-01-15_16-50-48/`

Update the `OUT=` variable in `run_ep<NNN>.sh` before each new experiment.

## Licensing
Source code is Apache 2.0. Model weights use NVIDIA-OneWay-Noncommercial-License.
