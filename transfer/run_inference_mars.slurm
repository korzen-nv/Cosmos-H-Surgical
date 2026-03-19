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
LUSTRE_USER=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk
WORK_DIR=$LUSTRE_USER/cosmos/Cosmos-H-Surgical/transfer
UV_INSTALL_DIR=$LUSTRE_USER/.local/uv
UV_CACHE=$LUSTRE_USER/.uv_cache
HF_CACHE=$LUSTRE_USER/.cache/huggingface

export UV_CACHE_DIR=$UV_CACHE
export HF_HOME=$HF_CACHE
export XDG_CACHE_HOME=$LUSTRE_USER/.cache
export PATH=$UV_INSTALL_DIR/bin:$LUSTRE_USER/.local/bin:$PATH

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

# --- Install uv if needed ---
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    mkdir -p $UV_INSTALL_DIR
    curl -LsSf https://astral.sh/uv/install.sh | CARGO_HOME=$UV_INSTALL_DIR sh
    export PATH=$UV_INSTALL_DIR/bin:$PATH
fi
echo "uv version: $(uv --version)"

# --- Install Python 3.10+ via uv (compute nodes have 3.8) ---
echo "=== Setting up Python ==="
uv python install 3.12
echo "Python: $(uv python find 3.12)"
echo ""

# --- Install dependencies (first run only, cached after) ---
echo "=== Installing dependencies ==="
uv sync --extra=cu128
echo "Install complete."
echo ""

# --- Environment info ---
echo "=== Environment ==="
uv run --no-sync python --version
uv run --no-sync python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPUs {torch.cuda.device_count()}')"
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
