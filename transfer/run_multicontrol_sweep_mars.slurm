#!/bin/bash
#SBATCH --job-name=cosmos_sweep
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --array=0-7%2
#SBATCH --partition=interactive,batch_block1,batch_block3,batch_block4
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

TASK_ID=$SLURM_ARRAY_TASK_ID
TOTAL_SPECS=$(wc -l < "$INPUT_JSONL")

echo "=== Sweep Task $TASK_ID (Job $SLURM_ARRAY_JOB_ID) on $(hostname) ==="
echo "Input: $INPUT_JSONL ($TOTAL_SPECS specs)"
echo "Output: $OUTPUT_DIR"
echo "Control: $CONTROL"
echo "This task: node $TASK_ID of 8, GPUs $((TASK_ID * 8))-$((TASK_ID * 8 + 7))"
echo ""

# --- Create run script (on Lustre so compute node can see it) ---
SWEEP_SH=$WORK_DIR/logs/cosmos_sweep_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.sh
cat > "$SWEEP_SH" << 'SWEEP_SCRIPT'
#!/bin/bash
set -euo pipefail

INPUT_JSONL=$1
OUTPUT_DIR=$2
CONTROL=$3
TOTAL_SPECS=$4
TASK_ID=$5

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

echo "=== Container on $(hostname), Task $TASK_ID ==="
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

# === Precompute edge + vis maps (once, with file lock) ===
EDGE_MAP=inputs/open_h/nonexpert-2026-01-15_16-50-48/edge/episode_000043.mp4
VIS_MAP=inputs/open_h/nonexpert-2026-01-15_16-50-48/vis/episode_000043.mp4
if [ ! -f "$EDGE_MAP" ] || [ ! -f "$VIS_MAP" ]; then
    LOCK="$OUTPUT_DIR/.precompute_lock"
    if mkdir "$LOCK" 2>/dev/null; then
        echo "=== Precomputing control maps (task $TASK_ID won lock) ==="
        mkdir -p "$OUTPUT_DIR"
        CUDA_VISIBLE_DEVICES=0 python inputs/open_h/nonexpert-2026-01-15_16-50-48/precompute_controls.py
        echo "Precompute done."
    else
        echo "=== Waiting for precomputation ==="
        while [ ! -f "$EDGE_MAP" ] || [ ! -f "$VIS_MAP" ]; do
            sleep 2
        done
        echo "Control maps found."
    fi
else
    echo "Control maps already exist, skipping precompute."
fi
echo ""

# === Parallel sweep — 8 GPUs on this node ===
echo "=== Running sweep (task $TASK_ID) ==="
mkdir -p "$OUTPUT_DIR"

GPUS_PER_NODE=8
SPECS_PER_GPU=4

for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
    GPU_GLOBAL=$((TASK_ID * GPUS_PER_NODE + gpu))
    START=$((GPU_GLOBAL * SPECS_PER_GPU + 1))
    END=$((START + SPECS_PER_GPU - 1))

    if [ "$START" -gt "$TOTAL_SPECS" ]; then
        echo "GPU $gpu (global $GPU_GLOBAL): no specs, skipping."
        continue
    fi

    TEMP_JSONL="/tmp/sweep_gpu${gpu}.jsonl"
    sed -n "${START},${END}p" "$INPUT_JSONL" > "$TEMP_JSONL"
    NUM_LINES=$(wc -l < "$TEMP_JSONL")

    echo "GPU $gpu (global $GPU_GLOBAL): specs $START-$END ($NUM_LINES lines)"

    CUDA_VISIBLE_DEVICES=$gpu \
    MASTER_PORT=$((29500 + GPU_GLOBAL)) \
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    python examples/inference.py \
        -i "$TEMP_JSONL" \
        -o "$OUTPUT_DIR" \
        "$CONTROL" \
        > "$OUTPUT_DIR/gpu_${GPU_GLOBAL}.log" 2>&1 &
done

echo "All GPU processes launched, waiting..."
wait
echo ""

echo "=== Task $TASK_ID done ==="
ls "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l | xargs -I{} echo "Videos in output dir: {}"
SWEEP_SCRIPT

chmod +x "$SWEEP_SH"

# --- Run in container ---
srun \
    --container-image="$CONTAINER" \
    --container-mounts="$WORK_DIR:/workspace,$HF_CACHE:/root/.cache/huggingface,$PIP_CACHE:/root/.cache/pip,$SWEEP_SH:/tmp/cosmos_sweep.sh" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    bash /tmp/cosmos_sweep.sh "$INPUT_JSONL" "$OUTPUT_DIR" "$CONTROL" "$TOTAL_SPECS" "$TASK_ID"
