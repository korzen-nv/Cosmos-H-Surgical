# Inference Performance & Optimization

## Observed performance (NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM, bf16)
- Single edge inference: **35 denoising steps × ~16 s/step ≈ 9.5 min/video** (1920×1080, 50 frames)
- GPU sits idle between episodes (sequential, batch_size=1)

## VRAM footprint per process (single-control, e.g. edge)
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

## Potential optimizations (not yet implemented)

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
