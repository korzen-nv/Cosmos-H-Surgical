#!/bin/bash
#SBATCH --job-name=cosmos_sweep
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=04:00:00
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch_block1,batch_block3,batch_block4
#SBATCH --account=healthcareeng_holoscan

set -euo pipefail

# --- Paths ---
LUSTRE_USER=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk
WORK_DIR=$LUSTRE_USER/cosmos/Cosmos-H-Surgical/transfer
PIP_CACHE=$LUSTRE_USER/.pip_cache
HF_CACHE=$LUSTRE_USER/.cache/huggingface
CONTAINER=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/user_data/lzbinden/containers/im4-onelogger.sqsh

# --- Args (defaults) ---
INPUT_JSONL=${1:-inputs/open_h/nonexpert-2026-01-15_16-50-48/ep043_sweep_all.jsonl}
OUTPUT_DIR=${2:-outputs/open_h/00_multicontrol_sweep/nonexpert-2026-01-15_16-50-48}
CONTROL=${3:-depth}

cd $WORK_DIR
mkdir -p logs

echo "=== Sweep Job $SLURM_JOB_ID on $(hostname) ==="
echo "Nodes: $SLURM_NNODES, GPUs/node: 8"
echo "Input: $INPUT_JSONL"
echo "Output: $OUTPUT_DIR"
echo "Control subcommand: $CONTROL"
TOTAL_SPECS=$(wc -l < "$INPUT_JSONL")
echo "Total specs: $TOTAL_SPECS"
echo ""

# --- Create install-and-run script ---
cat > /tmp/cosmos_sweep.sh << 'SWEEP_SCRIPT'
#!/bin/bash
set -euo pipefail

INPUT_JSONL=$1
OUTPUT_DIR=$2
CONTROL=$3
TOTAL_SPECS=$4

# Set HF token from cached credentials
if [ -f /root/.cache/huggingface/token ]; then
    export HF_TOKEN=$(cat /root/.cache/huggingface/token)
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
    echo "HF_TOKEN set from cache (${#HF_TOKEN} chars)"
    mkdir -p ~/.cache/huggingface
    echo "$HF_TOKEN" > ~/.cache/huggingface/token
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>/dev/null || true
else
    echo "WARNING: No HF token found at /root/.cache/huggingface/token"
fi

echo "=== Container on $(hostname), SLURM_NODEID=$SLURM_NODEID ==="
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')"
echo ""

# === Install dependencies ===
echo "=== Installing dependencies ==="
pip install hatchling hatch-vcs uv 2>&1 | tail -1
pip install --no-deps packages/* 2>&1 | tail -1
pip install --no-deps -e . 2>&1 | tail -1

pip install --no-deps \
    'megatron-core>=0.12.0,<0.14' \
    'diffusers>=0.35.2' \
    'peft>=0.17.1' \
    'timm>=1.0.24' \
    'sam2>=1.1.0' \
    'transformers==4.51.3' \
    'deepspeed>=0.16.0'

pip install \
    'cattrs>=25.2.0' \
    'easydict>=1.9' \
    'moderngl>=5.11.1' \
    'natsort>=8.4.0' \
    'OpenEXR>=3.3.1' \
    'pyarrow>=21.0.0' \
    'shapely>=2.0.5' \
    'albumentations>=2.0.8' \
    'attrs>=25.4.0' \
    'av>=16.0.1' \
    'better-profanity>=0.7.0' \
    'click>=8.3.0' \
    'einops>=0.8.1' \
    'ftfy>=6.3.1' \
    'huggingface-hub>=0.36.0' \
    'imageio[pyav,ffmpeg]>=2.37.0' \
    'loguru>=0.7.3' \
    'mediapy>=1.2.4' \
    'ml-dtypes>=0.5.3' \
    'numpy<2' \
    'numpydantic>=1.7.0' \
    'omegaconf>=2.3.0' \
    'opencv-python>=4.11.0.86' \
    'pandas>=2.3.0' \
    'pillow>=11.3.0' \
    'pydantic>=2.11.7' \
    'safetensors>=0.6.2' \
    'scikit-image>=0.25.2' \
    'scipy>=1.15.3' \
    'sentencepiece>=0.2.1' \
    'tqdm>=4.67.1' \
    'tyro>=0.9.35' \
    'tokenizers>=0.21,<0.22' \
    'decord>=0.6.0' \
    'retinaface-py>=0.0.2' \
    'qwen-vl-utils>=0.0.14' \
    'webdataset>=1.0.2' \
    'wandb>=0.22.2' \
    'multi-storage-client>=0.40.0' \
    2>&1 | tail -5

pip install 'video-depth-anything @ git+https://github.com/jeanachoi/Video-Depth-Anything.git' 2>&1 | tail -3 || echo "video-depth-anything failed (optional)"
echo "Install complete."
echo ""

python -c "import cosmos_transfer2; print('cosmos_transfer2 imported OK')"
echo ""

# === Phase 1: Precompute edge + vis maps (node 0 only) ===
PRECOMPUTE_FLAG="$OUTPUT_DIR/.precompute_done"
if [ "$SLURM_NODEID" -eq 0 ]; then
    echo "=== Precomputing control maps (node 0) ==="
    mkdir -p "$OUTPUT_DIR"
    CUDA_VISIBLE_DEVICES=0 python inputs/open_h/nonexpert-2026-01-15_16-50-48/precompute_controls.py
    touch "$PRECOMPUTE_FLAG"
    echo "Precompute done."
else
    echo "=== Waiting for precomputation (node $SLURM_NODEID) ==="
    while [ ! -f "$PRECOMPUTE_FLAG" ]; do
        sleep 2
    done
    echo "Precompute flag found, proceeding."
fi
echo ""

# === Phase 2: Parallel sweep ===
echo "=== Running sweep (node $SLURM_NODEID) ==="
mkdir -p "$OUTPUT_DIR"

GPUS_PER_NODE=8
SPECS_PER_GPU=4

for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
    GPU_GLOBAL=$((SLURM_NODEID * GPUS_PER_NODE + gpu))
    START=$((GPU_GLOBAL * SPECS_PER_GPU + 1))
    END=$((START + SPECS_PER_GPU - 1))

    # Check if we have specs for this GPU
    if [ "$START" -gt "$TOTAL_SPECS" ]; then
        echo "GPU $gpu (global $GPU_GLOBAL): no specs to process, skipping."
        continue
    fi

    # Extract this GPU's specs
    TEMP_JSONL="/tmp/sweep_gpu${gpu}.jsonl"
    sed -n "${START},${END}p" "$INPUT_JSONL" > "$TEMP_JSONL"
    NUM_LINES=$(wc -l < "$TEMP_JSONL")

    echo "GPU $gpu (global $GPU_GLOBAL): specs $START-$END ($NUM_LINES lines)"

    CUDA_VISIBLE_DEVICES=$gpu python examples/inference.py \
        -i "$TEMP_JSONL" \
        -o "$OUTPUT_DIR" \
        "$CONTROL" \
        > "$OUTPUT_DIR/gpu_${GPU_GLOBAL}.log" 2>&1 &
done

echo "All GPU processes launched, waiting..."
wait
echo ""

echo "=== Node $SLURM_NODEID done ==="
ls "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l | xargs -I{} echo "Videos in output dir: {}"
SWEEP_SCRIPT

chmod +x /tmp/cosmos_sweep.sh

# --- Run in container (all nodes) ---
srun \
    --container-image="$CONTAINER" \
    --container-mounts="$WORK_DIR:/workspace,$HF_CACHE:/root/.cache/huggingface,$PIP_CACHE:/root/.cache/pip,/tmp/cosmos_sweep.sh:/tmp/cosmos_sweep.sh" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    bash /tmp/cosmos_sweep.sh "$INPUT_JSONL" "$OUTPUT_DIR" "$CONTROL" "$TOTAL_SPECS"
