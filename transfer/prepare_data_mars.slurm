#!/bin/bash
#SBATCH --job-name=atlas120k_prep
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=16
#SBATCH --partition=interactive,batch_block1,batch_block3,batch_block4
#SBATCH --account=healthcareeng_holoscan

set -euo pipefail

# --- Paths ---
LUSTRE_USER=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk
WORK_DIR=$LUSTRE_USER/cosmos/Cosmos-H-Surgical/transfer
PIP_CACHE=$LUSTRE_USER/.pip_cache
ATLAS120K=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/atlas120k/atlas120k
OUTPUT_DIR=$LUSTRE_USER/cosmos/datasets/atlas120k_openh
CONTAINER=$LUSTRE_USER/cosmos/containers/cosmos-h-surgical-transfer.sqsh
BASE_CONTAINER=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/user_data/lzbinden/containers/im4-onelogger.sqsh

if [ ! -f "$CONTAINER" ]; then
    CONTAINER=$BASE_CONTAINER
fi

cd $WORK_DIR
mkdir -p logs

echo "=== Data Prep Job $SLURM_JOB_ID on $(hostname) ==="
echo "Input:  $ATLAS120K"
echo "Output: $OUTPUT_DIR"

# --- Create prep script ---
cat > /tmp/atlas120k_prep.sh << 'PREP_SCRIPT'
#!/bin/bash
set -euo pipefail

ATLAS120K=$1
OUTPUT_DIR=$2

echo "=== Installing minimal deps ==="
pip install opencv-python-headless pillow numpy 2>&1 | tail -3

echo ""
echo "=== Running data preparation ==="
python scripts/prepare_atlas120k_openh.py \
    --data-root "$ATLAS120K" \
    --output-root "$OUTPUT_DIR" \
    --splits train,val \
    --min-frames 93 \
    --fps 16 \
    --width 1280 \
    --height 720 \
    --workers 8

echo ""
echo "=== Verifying output ==="
echo "Train videos: $(ls $OUTPUT_DIR/train/*.mp4 2>/dev/null | grep -v seg | wc -l)"
echo "Train seg:    $(ls $OUTPUT_DIR/train/*.seg.mp4 2>/dev/null | wc -l)"
echo "Val videos:   $(ls $OUTPUT_DIR/val/*.mp4 2>/dev/null | grep -v seg | wc -l)"
echo "Val seg:      $(ls $OUTPUT_DIR/val/*.seg.mp4 2>/dev/null | wc -l)"
echo "Total size:   $(du -sh $OUTPUT_DIR)"
echo ""
echo "=== Done ==="
PREP_SCRIPT

chmod +x /tmp/atlas120k_prep.sh

# --- Run in container ---
srun \
    --container-image="$CONTAINER" \
    --container-mounts="$WORK_DIR:/workspace,$ATLAS120K:$ATLAS120K,$OUTPUT_DIR:$OUTPUT_DIR,$PIP_CACHE:/root/.cache/pip,/tmp/atlas120k_prep.sh:/tmp/atlas120k_prep.sh" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    bash /tmp/atlas120k_prep.sh "$ATLAS120K" "$OUTPUT_DIR"
