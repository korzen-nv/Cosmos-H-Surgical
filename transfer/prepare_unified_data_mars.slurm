#!/bin/bash
#SBATCH --job-name=openh_unified_prep
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=interactive,batch_block1,batch_block3,batch_block4
#SBATCH --account=healthcareeng_holoscan

set -euo pipefail

# --- Paths ---
LUSTRE_USER=/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk
LUSTRE_PROJ=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan
WORK_DIR=$LUSTRE_USER/cosmos/Cosmos-H-Surgical/transfer
PIP_CACHE=$LUSTRE_USER/.pip_cache
DATASETS=$LUSTRE_USER/cosmos/datasets
OUTPUT_DIR=$DATASETS/openh_unified

# Dataset paths on the cluster
ATLAS120K=$LUSTRE_PROJ/datasets/atlas120k/atlas120k
CHOLECSEG8K=$DATASETS/cholecseg8k
ENDOSCAPES=$DATASETS/endoscapes/endoscapes
HEISURF=$DATASETS/HeiSurf/HeiSurf

CONTAINER=$LUSTRE_USER/cosmos/containers/cosmos-h-surgical-transfer.sqsh
BASE_CONTAINER=$LUSTRE_PROJ/user_data/lzbinden/containers/im4-onelogger.sqsh
if [ ! -f "$CONTAINER" ]; then
    CONTAINER=$BASE_CONTAINER
fi

cd $WORK_DIR
mkdir -p logs $OUTPUT_DIR

echo "=== Unified Data Prep Job $SLURM_JOB_ID on $(hostname) ==="
echo "Output: $OUTPUT_DIR"
echo ""

# --- Create prep script ---
cat > /tmp/openh_unified_prep.sh << 'PREP_SCRIPT'
#!/bin/bash
set -euo pipefail

OUTPUT_DIR=$1
ATLAS120K=$2
CHOLECSEG8K=$3
ENDOSCAPES=$4
HEISURF=$5

echo "=== Installing minimal deps ==="
pip install opencv-python-headless pillow numpy 2>&1 | tail -3

echo ""
echo "=== Running unified data preparation ==="
python scripts/prepare_openh_unified.py \
    --output-root "$OUTPUT_DIR" \
    --atlas120k "$ATLAS120K" \
    --cholecseg8k "$CHOLECSEG8K" \
    --endoscapes "$ENDOSCAPES" \
    --heisurf "$HEISURF" \
    --min-frames 20 \
    --target-frames 93 \
    --width 1280 \
    --fps 16 \
    --workers 8

echo ""
echo "=== Verifying output ==="
echo "Train videos: $(ls $OUTPUT_DIR/train/*.mp4 2>/dev/null | grep -cv seg || echo 0)"
echo "Train seg:    $(ls $OUTPUT_DIR/train/*.seg.mp4 2>/dev/null | wc -l)"
echo "Total size:   $(du -sh $OUTPUT_DIR)"
echo ""
echo "=== Done ==="
PREP_SCRIPT

chmod +x /tmp/openh_unified_prep.sh

# --- Run in container ---
srun \
    --container-image="$CONTAINER" \
    --container-mounts="$WORK_DIR:/workspace,$ATLAS120K:$ATLAS120K,$CHOLECSEG8K:$CHOLECSEG8K,$ENDOSCAPES:$ENDOSCAPES,$HEISURF:$HEISURF,$OUTPUT_DIR:$OUTPUT_DIR,$PIP_CACHE:/root/.cache/pip,/tmp/openh_unified_prep.sh:/tmp/openh_unified_prep.sh" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    bash /tmp/openh_unified_prep.sh "$OUTPUT_DIR" "$ATLAS120K" "$CHOLECSEG8K" "$ENDOSCAPES" "$HEISURF"
