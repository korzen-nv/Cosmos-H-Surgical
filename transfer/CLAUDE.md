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

## Inference Performance & Optimization

### Observed performance (NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM, bf16)
- Single edge inference: **35 denoising steps × ~16 s/step ≈ 9.5 min/video** (1920×1080, 50 frames)
- GPU sits idle between episodes (sequential, batch_size=1)

### VRAM footprint per process (single-control, e.g. edge)
| Component | VRAM |
|---|---|
| Edge transfer model (control + base DiT) | ~8.3 GB |
| VAE tokenizer (Wan2.1) | ~0.5 GB |
| Cosmos-Reason1-7B text encoder | ~15.6 GB |
| Activations (1920×1080 × 50 frames) | ~8–12 GB |
| **Estimated peak** | **~33–36 GB** |
| **Measured peak** | **54,230 MiB (~53 GB)** |

Note: even for single-control inference all four control checkpoints are downloaded to disk on first run (edge/depth/seg/vis, ~4.4 GB each), but only the matching one is loaded into VRAM.

**Parallelism implication:** At 53 GB per process, two instances would require ~106 GB — exceeding the 96 GB available. Parallel processes are therefore **not feasible** without first implementing text-encoder CPU offloading (frees ~15 GB → ~38 GB/process → 2 instances fit in ~76 GB).

### Potential optimizations (not yet implemented)

**1. Multiple parallel processes (requires text-encoder offloading first)**
Measured peak is 53 GB/process — two instances need ~106 GB, exceeding 96 GB. Text-encoder CPU offloading (#6 below) reduces this to ~38 GB/process, making 2 parallel processes feasible (~76 GB). Expected ~1.5–1.7× wall-clock improvement once offloading is in place.

**2. Increased batch_size (code change in inference_pipeline.py)**
`self.batch_size = 1` is hardcoded in `ControlVideo2WorldInference.__init__`. Bumping to 2 shares the ~24 GB of model weights across two videos and roughly doubles activation memory (~53 GB total). True ~1.8× GPU throughput improvement.

**3. torch.compile (low-effort, zero quality loss)**
`use_torch_compile` flag exists in the model config (`_src/predict2/models/text2world_model_rectified_flow.py`) but is not wired into the inference pipeline. Enabling it for the DiT forward pass would give ~20–30% step-time reduction on Blackwell.

**4. FP8 PTQ via NVIDIA ModelOpt (medium effort)**
The DiT runs in bf16. Applying post-training FP8 quantization to linear layers via `modelopt.torch.quantization` (calibration only, no retraining) would reduce step time to ~8–10 s on Blackwell's native FP8 tensor cores. NATTEN attention already detects Blackwell (arch 100/103) and has FP8 forward-pass support. Skip NATTEN layers in the quantization config; quantize only the FFN/projection linears.

**5. Distilled model variant**
`ModelKey(variant, distilled=True)` is registered in `MODEL_CHECKPOINTS`. If a distilled checkpoint is available it reduces denoising steps from 35 to ~8, giving ~4× speedup with small quality tradeoff.

**6. Text-encoder CPU offloading**
Cosmos-Reason1-7B (15.6 GB) is used only once per video to encode the prompt. Offloading it to CPU after encoding frees ~15 GB per process, enabling a third parallel process or larger batch sizes.

**7. NVFP4 (high effort)**
No infrastructure exists. Would require ModelOpt MX FP4 integration. Blackwell has native hardware support; potential 2–4× speedup but needs quality validation for diffusion mid-loop use.

## Datasets

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
