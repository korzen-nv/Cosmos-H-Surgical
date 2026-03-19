#!/bin/bash
#SBATCH --job-name=cosmos_transfer_inf
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=interactive,batch_block1,batch_block3,batch_block4
#SBATCH --account=healthcareeng_holoscan

set -euo pipefail

# --- Paths ---
WORK_DIR=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk/cosmos/Cosmos-H-Surgical/transfer
UV_CACHE=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk/.uv_cache
HF_CACHE=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk/.cache/huggingface

export UV_CACHE_DIR=$UV_CACHE
export HF_HOME=$HF_CACHE
export XDG_CACHE_HOME=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk/.cache

# --- Args (defaults) ---
INPUT_FILE=${1:-assets/multicontrol.jsonl}
OUTPUT_DIR=${2:-outputs/mars_test}
CONTROL=${3:-depth}
NUM_GPUS=${4:-8}

cd $WORK_DIR
mkdir -p logs

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo "Control: $CONTROL"
echo "GPUs: $NUM_GPUS"
echo ""

# --- Check CUDA ---
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# --- Check Python/CUDA versions ---
echo "=== Environment ==="
python3 --version
nvcc --version 2>/dev/null | tail -1 || echo "nvcc not found"
echo ""

# --- Install uv if needed ---
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    pip install --quiet uv
fi
echo "uv version: $(uv --version)"

# --- Install dependencies (first run only, cached after) ---
echo "=== Installing dependencies ==="
uv python install
uv sync --extra=cu128
echo "Install complete."
echo ""

# --- Run inference ---
echo "=== Running inference ==="
mkdir -p $OUTPUT_DIR

uv run --no-sync torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    examples/inference.py \
    -i $INPUT_FILE \
    -o $OUTPUT_DIR \
    $CONTROL

echo ""
echo "=== Done ==="
echo "Output: $OUTPUT_DIR"
ls -la $OUTPUT_DIR/
