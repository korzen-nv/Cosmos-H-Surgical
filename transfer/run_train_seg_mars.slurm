#!/bin/bash
#SBATCH --job-name=cosmos_seg_openh
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=batch_block1,batch_block3,batch_block4
#SBATCH --account=healthcareeng_holoscan

set -euo pipefail

# --- Paths ---
LUSTRE_USER=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk
WORK_DIR=$LUSTRE_USER/cosmos/Cosmos-H-Surgical/transfer
PIP_CACHE=$LUSTRE_USER/.pip_cache
HF_CACHE=$LUSTRE_USER/.cache/huggingface
DATASET_DIR=$LUSTRE_USER/cosmos/datasets/atlas120k_openh/train
JSON_PATH=$LUSTRE_USER/cosmos/datasets/atlas120k_openh/train.json
CKPT_DIR=$LUSTRE_USER/cosmos/checkpoints
CONTAINER=$LUSTRE_USER/cosmos/containers/cosmos-h-surgical-transfer.sqsh
BASE_CONTAINER=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/user_data/lzbinden/containers/im4-onelogger.sqsh

if [ ! -f "$CONTAINER" ]; then
    CONTAINER=$BASE_CONTAINER
fi

# --- Training hyperparameters ---
MAX_ITER=${1:-5000}
LR=${2:-0.0000432}         # 2^(-14.5), same as original seg fine-tuning
SAVE_ITER=${3:-500}

cd $WORK_DIR
mkdir -p logs $CKPT_DIR

echo "=== Training Job $SLURM_JOB_ID on $(hostname) ==="
echo "Dataset:  $DATASET_DIR"
echo "Max iter: $MAX_ITER"
echo "LR:       $LR"
echo "Save:     every $SAVE_ITER iters"
echo ""

# --- GPU Info ---
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# --- Create training script ---
cat > /tmp/cosmos_train.sh << 'TRAIN_SCRIPT'
#!/bin/bash
set -euo pipefail

DATASET_DIR=$1
JSON_PATH=$2
CKPT_DIR=$3
MAX_ITER=$4
LR=$5
SAVE_ITER=$6

# Set HF token
if [ -f /root/.cache/huggingface/token ]; then
    export HF_TOKEN=$(cat /root/.cache/huggingface/token)
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
    echo "HF_TOKEN set (${#HF_TOKEN} chars)"
else
    echo "WARNING: No HF token found"
fi

echo "=== Container Environment ==="
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')"
echo ""

# Install packages (same as inference slurm)
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

# Verify
python -c "import cosmos_transfer2; print('cosmos_transfer2 imported OK')"

# Download seg checkpoint from HuggingFace if not cached
SEG_CKPT="$CKPT_DIR/cosmos-h-surgical-transfer-seg_model_ema_bf16.pt"
if [ ! -f "$SEG_CKPT" ]; then
    echo "=== Downloading seg checkpoint from HuggingFace ==="
    python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='nvidia/Cosmos-H-Surgical',
    filename='transfer/seg/cosmos-h-surgical-transfer-seg_model_ema_bf16.pt',
    local_dir='$CKPT_DIR',
    local_dir_use_symlinks=False,
)
print(f'Downloaded to: {path}')
"
    # Move from nested path to flat location
    if [ -f "$CKPT_DIR/transfer/seg/cosmos-h-surgical-transfer-seg_model_ema_bf16.pt" ]; then
        mv "$CKPT_DIR/transfer/seg/cosmos-h-surgical-transfer-seg_model_ema_bf16.pt" "$SEG_CKPT"
        rm -rf "$CKPT_DIR/transfer"
    fi
    echo "Checkpoint ready: $(ls -lh $SEG_CKPT)"
else
    echo "Seg checkpoint found: $(ls -lh $SEG_CKPT)"
fi
echo ""

# Set distributed training env
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12341
export NCCL_DEBUG=WARN

echo "=== Starting training ==="
echo "  Nodes:    $SLURM_NNODES"
echo "  GPUs:     8"
echo "  Dataset:  $DATASET_DIR"
echo "  Checkpoint: $SEG_CKPT"
echo "  Max iter: $MAX_ITER"
echo "  LR:       $LR"
echo ""

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
    -- \
    experiment=vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow \
    data_train=surgical_singleview_train_data_seg \
    job.group=cosmos_h_surgical_seg_openh \
    job.name=seg_openh_atlas120k_finetune \
    model.config.hint_keys=seg \
    model.config.fsdp_shard_size=8 \
    model_parallel.context_parallel_size=1 \
    dataloader_train.dataset.dataset_dir=$DATASET_DIR \
    dataloader_train.dataset.json_path=$JSON_PATH \
    dataloader_train.dataset.enlarged_factor=10.0 \
    dataloader_train.dataset.hint_key=control_input_seg \
    checkpoint.load_path=$SEG_CKPT \
    checkpoint.load_training_state=False \
    checkpoint.strict_resume=False \
    checkpoint.save_iter=$SAVE_ITER \
    checkpoint.save_to_object_store.enabled=False \
    checkpoint.load_from_object_store.enabled=False \
    optimizer.lr=$LR \
    trainer.max_iter=$MAX_ITER \
    trainer.logging_iter=50 \
    trainer.run_validation=False \
    job.wandb_mode=offline

echo ""
echo "=== Training complete ==="
echo "Checkpoints:"
ls -la $CKPT_DIR/ 2>/dev/null || echo "Check job output dir for checkpoints"
TRAIN_SCRIPT

chmod +x /tmp/cosmos_train.sh

# --- Run in container ---
srun \
    --container-image="$CONTAINER" \
    --container-mounts="$WORK_DIR:/workspace,$HF_CACHE:/root/.cache/huggingface,$PIP_CACHE:/root/.cache/pip,$DATASET_DIR:$DATASET_DIR,$JSON_PATH:$JSON_PATH,$CKPT_DIR:$CKPT_DIR,/tmp/cosmos_train.sh:/tmp/cosmos_train.sh" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    bash /tmp/cosmos_train.sh "$DATASET_DIR" "$JSON_PATH" "$CKPT_DIR" "$MAX_ITER" "$LR" "$SAVE_ITER"
