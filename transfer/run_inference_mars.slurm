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
# Existing container (PyTorch 2.5, CUDA 12.6, flash_attn, Python 3.10)
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

# --- Create install script (avoids quoting issues in srun bash -c) ---
cat > /tmp/cosmos_install_and_run.sh << 'INSTALL_SCRIPT'
#!/bin/bash
set -euo pipefail

INPUT_FILE=$1
OUTPUT_DIR=$2
CONTROL=$3
NUM_GPUS=$4

echo "=== Container Environment ==="
python --version
nvcc --version 2>/dev/null | tail -1 || echo "no nvcc"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')"
echo ""

# Install workspace packages (--no-deps to avoid cosmos-oss==0.1.0 conflict)
echo "=== Installing dependencies ==="
pip install hatchling hatch-vcs 2>&1 | tail -1
pip install --no-deps packages/* 2>&1 | tail -1
pip install --no-deps -e . 2>&1 | tail -1

# Install actual deps from pyproject.toml (>= pins resolve for Python 3.10)
# Skip: cosmos-oss (installed above), torch/nvidia (in container)
# CRITICAL: Use --no-deps to prevent transitive deps from upgrading torch
# Then install only the light deps that don't pull in torch
pip install --no-deps \
    'megatron-core>=0.12.0' \
    'diffusers>=0.35.2' \
    'peft>=0.17.1' \
    'timm>=1.0.24' \
    'sam2>=1.1.0' \
    'transformers==4.51.3'

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

# Verify
python -c "import cosmos_transfer2; print('cosmos_transfer2 imported OK')"
echo ""

# Run inference
echo "=== Running inference ==="
mkdir -p $OUTPUT_DIR

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    examples/inference.py \
    -i $INPUT_FILE \
    -o $OUTPUT_DIR \
    $CONTROL

echo ""
echo "=== Done ==="
ls -la $OUTPUT_DIR/
INSTALL_SCRIPT

chmod +x /tmp/cosmos_install_and_run.sh

# --- Run in container ---
srun \
    --container-image="$CONTAINER" \
    --container-mounts="$WORK_DIR:/workspace,$HF_CACHE:/root/.cache/huggingface,$PIP_CACHE:/root/.cache/pip,/tmp/cosmos_install_and_run.sh:/tmp/cosmos_install_and_run.sh" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    bash /tmp/cosmos_install_and_run.sh "$INPUT_FILE" "$OUTPUT_DIR" "$CONTROL" "$NUM_GPUS"
