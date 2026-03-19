#!/bin/bash
#SBATCH --job-name=cosmos_build_sqsh
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=interactive,batch_block1,batch_block3,batch_block4
#SBATCH --account=healthcareeng_holoscan

set -euo pipefail

LUSTRE_USER=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk
WORK_DIR=$LUSTRE_USER/cosmos/Cosmos-H-Surgical/transfer
CONTAINER_DIR=$LUSTRE_USER/cosmos/containers
NGC_IMAGE="nvcr.io#nvidia/pytorch:25.04-py3"
SQSH_FILE=$CONTAINER_DIR/pytorch-25.04-py3.sqsh

mkdir -p $CONTAINER_DIR logs

echo "=== Building container ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"
echo "Image: $NGC_IMAGE"
echo "Output: $SQSH_FILE"
echo ""

if [ -f "$SQSH_FILE" ]; then
    echo "Container already exists: $SQSH_FILE"
    echo "Delete it first if you want to rebuild."
else
    echo "Importing NGC image (this may take 10-20 min)..."
    srun enroot import -o $SQSH_FILE -- docker://$NGC_IMAGE
    echo "Import complete: $(ls -lh $SQSH_FILE)"
fi

echo ""
echo "=== Testing container ==="
srun \
    --container-image="$SQSH_FILE" \
    --container-workdir=/workspace \
    --container-mounts="$WORK_DIR:/workspace,$LUSTRE_USER/.cache:/root/.cache" \
    --no-container-mount-home \
    bash -c "
python --version
pip --version
python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}'\"
echo 'Container test OK'
"
