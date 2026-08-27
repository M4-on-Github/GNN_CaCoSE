#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Download the datasets into /data/$USER/CaCoSE/datasets.
#
#   sbatch slurm/prefetch.sh
#
# Run as a job because apptainer is not installed on head1. No GPU, a few minutes.
#
# If this fails with a network error, the compute nodes have no outbound access and the
# datasets have to reach /data another way -- see the note at the bottom of this file.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH -J cacose-prefetch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/data/%u/CaCoSE/logs/cacose-prefetch_%j.out
#SBATCH --error=/data/%u/CaCoSE/logs/cacose-prefetch_%j.err

set -e
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

DATA_DIR="/data/$USER/CaCoSE"
SIF="$DATA_DIR/containers/cacose.sif"
PYTHON=/opt/conda/bin/python3

echo "=========================================="
echo " Node    : $(hostname)"
echo " Started : $(date)"
echo " Target  : $DATA_DIR/datasets"
echo "=========================================="

mkdir -p "$DATA_DIR/datasets" "$DATA_DIR/logs"

if [ ! -f "$SIF" ]; then
    echo "ERROR: container not found at $SIF -- run: sbatch slurm/build.sh" >&2
    exit 1
fi

apptainer exec --containall \
    --pwd "$REPO" \
    --env USER="$USER" \
    --env CACOSE_DATA_ROOT="$DATA_DIR/datasets" \
    --env XDG_CACHE_HOME="$DATA_DIR/.cache" \
    --bind /tmp:/tmp \
    --bind "$REPO:$REPO" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" $PYTHON -m scripts.prefetch_data --all

echo "[$(date)] datasets ready under $DATA_DIR/datasets"
du -sh "$DATA_DIR/datasets"

# If the download failed because compute nodes have no outbound network:
#   1. Download on any machine that does have network (your laptop works):
#        python -m scripts.prefetch_data --all --data-root ./data
#   2. Copy it over:
#        scp -r ./data/* mmyatmau@head1.condo.cs.cmu.edu:/data/$USER/CaCoSE/datasets/
#   It is about 50 MB in total.
