#!/bin/bash
#SBATCH --job-name=cosmos_build
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=interactive,batch_block1,batch_block3,batch_block4
#SBATCH --account=healthcareeng_holoscan

set -euo pipefail

# --- Paths ---
LUSTRE_USER=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk
WORK_DIR=$LUSTRE_USER/cosmos/Cosmos-H-Surgical/transfer
PIP_CACHE=$LUSTRE_USER/.pip_cache
HF_CACHE=$LUSTRE_USER/.cache/huggingface
CONTAINER_DIR=$LUSTRE_USER/cosmos/containers

# Base container (PyTorch 2.5, CUDA 12.6, Python 3.10, flash_attn)
BASE_CONTAINER=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/user_data/lzbinden/containers/im4-onelogger.sqsh
OUTPUT_SQSH=$CONTAINER_DIR/cosmos-h-surgical-transfer.sqsh

cd $WORK_DIR
mkdir -p $CONTAINER_DIR logs

echo "=== Building cosmos-h-surgical-transfer container ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"
echo "Base: $BASE_CONTAINER"
echo "Output: $OUTPUT_SQSH"
echo ""

if [ -f "$OUTPUT_SQSH" ]; then
    echo "Container already exists: $OUTPUT_SQSH ($(ls -lh $OUTPUT_SQSH | awk '{print $5}'))"
    echo "Delete it first if you want to rebuild."
    echo ""
    echo "=== Testing existing container ==="
    srun \
        --container-image="$OUTPUT_SQSH" \
        --container-workdir=/workspace \
        --container-mounts="$WORK_DIR:/workspace,$HF_CACHE:/root/.cache/huggingface" \
        --no-container-mount-home \
        bash -c "
python --version
python -c 'import cosmos_transfer2; print(\"cosmos_transfer2 OK\")'
python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}\")'
echo 'Container test OK'
"
    exit 0
fi

# --- Create install script ---
cat > /tmp/cosmos_install.sh << 'INSTALL_SCRIPT'
#!/bin/bash
set -euo pipefail

echo "=== Container environment ==="
python --version
nvcc --version 2>/dev/null | tail -1 || echo "no nvcc"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')"
echo ""

# Set HF token
if [ -f /root/.cache/huggingface/token ]; then
    export HF_TOKEN=$(cat /root/.cache/huggingface/token)
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
    mkdir -p ~/.cache/huggingface
    echo "$HF_TOKEN" > ~/.cache/huggingface/token
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>/dev/null || true
    echo "HF_TOKEN set (${#HF_TOKEN} chars)"
fi
echo ""

echo "=== Installing workspace packages ==="
pip install hatchling hatch-vcs uv 2>&1 | tail -3
pip install --no-deps packages/* 2>&1 | tail -3
pip install --no-deps -e . 2>&1 | tail -3
echo ""

echo "=== Installing torch-dependent packages (--no-deps) ==="
pip install --no-deps \
    'megatron-core>=0.12.0,<0.14' \
    'diffusers>=0.35.2' \
    'peft>=0.17.1' \
    'timm>=1.0.24' \
    'sam2>=1.1.0' \
    'transformers==4.51.3' \
    'deepspeed>=0.16.0' \
    2>&1 | tail -5
echo ""

echo "=== Installing remaining dependencies ==="
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
echo ""

pip install 'video-depth-anything @ git+https://github.com/jeanachoi/Video-Depth-Anything.git' 2>&1 | tail -5 || echo "video-depth-anything failed (optional)"
echo ""

echo "=== Verifying ==="
python -c "import cosmos_transfer2; print('cosmos_transfer2 imported OK')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')"
python -c "import megatron; print(f'megatron-core OK')"
python -c "import diffusers; print(f'diffusers {diffusers.__version__} OK')"
python -c "import deepspeed; print(f'deepspeed {deepspeed.__version__} OK')"
echo ""
echo "=== Install complete ==="
INSTALL_SCRIPT

chmod +x /tmp/cosmos_install.sh

# --- Install deps in container and export as .sqsh ---
echo "=== Installing deps in container ==="
srun \
    --container-image="$BASE_CONTAINER" \
    --container-mounts="$WORK_DIR:/workspace,$HF_CACHE:/root/.cache/huggingface,$PIP_CACHE:/root/.cache/pip,/tmp/cosmos_install.sh:/tmp/cosmos_install.sh" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    --container-save="$OUTPUT_SQSH" \
    bash /tmp/cosmos_install.sh

echo ""
echo "=== Container saved ==="
ls -lh "$OUTPUT_SQSH"
echo ""

echo "=== Testing new container ==="
srun \
    --container-image="$OUTPUT_SQSH" \
    --container-workdir=/workspace \
    --container-mounts="$WORK_DIR:/workspace,$HF_CACHE:/root/.cache/huggingface" \
    --no-container-mount-home \
    bash -c "
python -c 'import cosmos_transfer2; print(\"cosmos_transfer2 OK\")'
python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}\")'
echo 'Container test OK'
"
