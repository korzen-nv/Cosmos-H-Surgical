#!/bin/bash
#SBATCH --job-name=seg_openh_sweep
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=interactive,batch_block1,batch_block3,batch_block4
#SBATCH --account=healthcareeng_holoscan

set -euo pipefail

# --- Paths ---
LUSTRE_USER=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk
WORK_DIR=$LUSTRE_USER/cosmos/Cosmos-H-Surgical/transfer
PIP_CACHE=$LUSTRE_USER/.pip_cache
HF_CACHE=$LUSTRE_USER/.cache/huggingface
CKPT_DIR=$LUSTRE_USER/cosmos/checkpoints

# Which training checkpoint to use
# Arg 1: either an iter name (e.g. iter_000000600) or a full path to the DCP checkpoint dir
CKPT_ARG=${1:-iter_000000600}
if [[ "$CKPT_ARG" == /* ]]; then
    TRAIN_CKPT_DIR=$CKPT_ARG
else
    TRAIN_CKPT_DIR=$CKPT_DIR/output/cosmos_transfer2/cosmos_h_surgical_seg_openh/seg_openh_atlas120k_finetune/checkpoints/$CKPT_ARG
fi

# Sweep input and output
INPUT_FILE=${2:-inputs/open_h/nonexpert-2026-01-15_16-50-48/ep043_seg_openh_cw_sweep.jsonl}
OUTPUT_DIR=${3:-outputs/open_h/seg_openh_sweep}

CONTAINER=$LUSTRE_USER/cosmos/containers/cosmos-h-surgical-transfer.sqsh
BASE_CONTAINER=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/user_data/lzbinden/containers/im4-onelogger.sqsh
if [ ! -f "$CONTAINER" ]; then
    CONTAINER=$BASE_CONTAINER
fi

cd $WORK_DIR
mkdir -p logs

echo "=== Seg OpenH Sweep Job $SLURM_JOB_ID on $(hostname) ==="
echo "Checkpoint: $TRAIN_CKPT_DIR"
echo "Input:      $INPUT_FILE"
echo "Output:     $OUTPUT_DIR"
echo ""

# --- Create run script ---
cat > /tmp/seg_openh_sweep.sh << 'SWEEP_SCRIPT'
#!/bin/bash
set -euo pipefail

CKPT_DIR=$1
TRAIN_CKPT_DIR=$2
INPUT_FILE=$3
OUTPUT_DIR=$4

# HF token
if [ -f /root/.cache/huggingface/token ]; then
    export HF_TOKEN=$(cat /root/.cache/huggingface/token)
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
    echo "HF_TOKEN set (${#HF_TOKEN} chars)"
fi

echo "=== Container Environment ==="
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')"
echo ""

# Install deps (same as training/inference)
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
    'natsort>=8.4.0' \
    'pyarrow>=21.0.0' \
    'attrs>=25.4.0' \
    'av>=16.0.1' \
    'click>=8.3.0' \
    'einops>=0.8.1' \
    'ftfy>=6.3.1' \
    'huggingface-hub>=0.36.0' \
    'imageio[pyav,ffmpeg]>=2.37.0' \
    'loguru>=0.7.3' \
    'numpy<2' \
    'omegaconf>=2.3.0' \
    'opencv-python>=4.11.0.86' \
    'pandas>=2.3.0' \
    'pillow>=11.3.0' \
    'pydantic>=2.11.7' \
    'safetensors>=0.6.2' \
    'scipy>=1.15.3' \
    'sentencepiece>=0.2.1' \
    'tqdm>=4.67.1' \
    'tyro>=0.9.35' \
    'tokenizers>=0.21,<0.22' \
    'decord>=0.6.0' \
    'qwen-vl-utils>=0.0.14' \
    'webdataset>=1.0.2' \
    'wandb>=0.22.2' \
    'multi-storage-client>=0.40.0' \
    2>&1 | tail -3
echo "Install complete."
echo ""

python -c "import cosmos_transfer2; print('cosmos_transfer2 imported OK')"

# === Step 1: Convert DCP checkpoint to consolidated .pt ===
# Use checkpoint-specific .pt name to avoid conflicts between different training runs
CKPT_BASENAME=$(basename "$TRAIN_CKPT_DIR")
PT_CKPT="$TRAIN_CKPT_DIR/model_ema_bf16.pt"
if [ ! -f "$PT_CKPT" ]; then
    echo "=== Converting DCP checkpoint to .pt ==="
    echo "  Input:  $TRAIN_CKPT_DIR/model"
    echo "  Output: $TRAIN_CKPT_DIR/"

    python scripts/convert_distcp_to_pt.py \
        "$TRAIN_CKPT_DIR/model" \
        "$TRAIN_CKPT_DIR/"

    echo "Converted checkpoint: $(ls -lh $PT_CKPT)"
else
    echo "Converted checkpoint found: $(ls -lh $PT_CKPT)"
fi
echo ""

# === Step 2: Run sweep — batch specs across 8 GPUs ===
echo "=== Running seg sweep ==="
mkdir -p $OUTPUT_DIR

INPUT_DIR=$(dirname "$INPUT_FILE")
NUM_LINES=$(wc -l < "$INPUT_FILE")
NUM_GPUS=8
NUM_BATCHES=$(( (NUM_LINES + NUM_GPUS - 1) / NUM_GPUS ))
echo "  $NUM_LINES samples, $NUM_BATCHES batch(es) of $NUM_GPUS GPUs"

for batch in $(seq 0 $((NUM_BATCHES - 1))); do
    echo ""
    echo "--- Batch $((batch + 1))/$NUM_BATCHES ---"
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        LINE=$(( batch * NUM_GPUS + gpu + 1 ))
        if [ $LINE -le $NUM_LINES ]; then
            GPU_SPEC="$INPUT_DIR/_sweep_gpu${gpu}.json"
            sed -n "${LINE}p" "$INPUT_FILE" > "$GPU_SPEC"
            echo "  GPU $gpu: $(python -c "import json; d=json.load(open('$GPU_SPEC')); print(d['name'])")"
            CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$((53900 + gpu)) python examples/inference.py \
                -i "$GPU_SPEC" \
                -o "$OUTPUT_DIR" \
                --checkpoint-path "$PT_CKPT" \
                seg &
        fi
    done
    wait
done

echo ""
echo "=== Sweep complete ==="
ls -la $OUTPUT_DIR/*.mp4 2>/dev/null | head -20
echo "Total outputs: $(ls $OUTPUT_DIR/*.mp4 2>/dev/null | wc -l)"
SWEEP_SCRIPT

chmod +x /tmp/seg_openh_sweep.sh

# --- Run in container ---
srun \
    --container-image="$CONTAINER" \
    --container-mounts="$WORK_DIR:/workspace,$HF_CACHE:/root/.cache/huggingface,$PIP_CACHE:/root/.cache/pip,$CKPT_DIR:$CKPT_DIR,/tmp/seg_openh_sweep.sh:/tmp/seg_openh_sweep.sh" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    bash /tmp/seg_openh_sweep.sh "$CKPT_DIR" "$TRAIN_CKPT_DIR" "$INPUT_FILE" "$OUTPUT_DIR"
