#!/bin/bash
#SBATCH --job-name=cosmos_transfer_inf
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
# Existing container from the old cosmos project (has PyTorch + CUDA)
CONTAINER=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/user_data/lzbinden/containers/im4-onelogger.sqsh

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

# --- GPU Info ---
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# --- Run in container ---
srun \
    --container-image="$CONTAINER" \
    --container-mounts="$WORK_DIR:/workspace,$HF_CACHE:/root/.cache/huggingface,$PIP_CACHE:/root/.cache/pip" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    bash -c "
set -euo pipefail

echo '=== Container Environment ==='
python --version
nvcc --version 2>/dev/null | tail -1 || echo 'no nvcc'
pip --version
echo ''

# Check what PyTorch we have
python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA avail={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}\")' || echo 'No PyTorch'
echo ''

# Install project deps
# Note: pyproject.toml uses uv workspace (cosmos-oss==0.1.0 override),
# so we install everything --no-deps, then install actual deps from requirements
# Container has PyTorch 2.5 + flash_attn + CUDA 12.6 - DO NOT overwrite these
echo '=== Installing dependencies ==='

# Filter nightly-requirements.txt:
# - Remove lines with platform_machine markers (old pip can't parse)
# - Remove packages that would overwrite container's PyTorch/CUDA stack
# - Remove packages requiring Python 3.11+ (container has 3.10)
python3 -c \"
import sys
skip_prefixes = [
    'torch==', 'torchvision==', 'torchaudio==', 'torchao==',
    'triton==', 'nvidia-', 'cuda-', 'flash-attn', 'flash_attn',
    'transformer-engine', 'apex',
    'decord @',  # aarch64 wheel URL
]
skip_exact = {'contourpy==1.3.3'}  # needs Python 3.11+
with open('docker/nightly-requirements.txt') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if 'platform_machine' in line:
            continue
        if any(line.startswith(p) for p in skip_prefixes):
            continue
        if line in skip_exact:
            # Use compatible version instead
            print('contourpy')
            continue
        print(line)
\" > /tmp/filtered-reqs.txt

echo \"Installing \$(wc -l < /tmp/filtered-reqs.txt) packages (keeping container PyTorch)...\"
pip install hatchling hatch-vcs
pip install --no-deps packages/*
pip install --no-deps -e .
pip install -r /tmp/filtered-reqs.txt
pip install decord
echo 'Install complete.'
echo ''

# Verify import
python -c 'import cosmos_transfer2; print(\"cosmos_transfer2 imported OK\")'
echo ''

# Run inference
echo '=== Running inference ==='
mkdir -p $OUTPUT_DIR

torchrun \\
    --nproc_per_node=$NUM_GPUS \\
    --nnodes=1 \\
    --node_rank=0 \\
    examples/inference.py \\
    -i $INPUT_FILE \\
    -o $OUTPUT_DIR \\
    $CONTROL

echo ''
echo '=== Done ==='
ls -la $OUTPUT_DIR/
"
