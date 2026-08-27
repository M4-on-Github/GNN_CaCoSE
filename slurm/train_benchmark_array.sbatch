#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Run one config across 10 seeds as a SLURM array.
#
#   sbatch slurm/train_benchmark_array.sbatch configs/cora.yaml
#
#   (normally submitted via scripts/submit_benchmark_sweep.sh, which also rebuilds a stale container)
#
# %3 (JobArrayTaskLimit) caps concurrency at three tasks: SLURM runs three and starts the next
# as each finishes, so the sweep never takes more than three GPUs at once.
#
# No --nodelist: peak usage is ~1-2 GB VRAM, so any node in the partition will do and the
# queue is shorter for it. cu121 in the container covers every GPU the cluster has.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH -J cacose
#SBATCH --array=0-9%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/%u/CaCoSE/logs/cacose_%A_%a.out
#SBATCH --error=/data/%u/CaCoSE/logs/cacose_%A_%a.err

set -e
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

CONFIG=${1:?usage: sbatch slurm/train_benchmark_array.sbatch <config.yaml>}
SEED=${SLURM_ARRAY_TASK_ID:-0}

DATA_DIR="/data/$USER/CaCoSE"
SIF="$DATA_DIR/containers/cacose.sif"

# Absolute, not `python`: apptainer exec does not inherit the Docker image's ENV PATH.
PYTHON=/opt/conda/bin/python3

# Resolved on the host: the container has no git installed, and every result records this so
# the numbers in REPRO_REPORT.md can be traced back to a commit.
GIT_SHA=$(git rev-parse --short=12 HEAD 2>/dev/null || echo "")

echo "=========================================="
echo " Job         : ${SLURM_ARRAY_JOB_ID:-local}_${SEED}"
echo " Node        : $(hostname)"
echo " GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo " Config      : $CONFIG"
echo " Seed        : $SEED"
echo " Repo        : $REPO"
echo " Output      : $DATA_DIR"
echo "=========================================="

if [ ! -f "$SIF" ]; then
    echo "ERROR: container not found at $SIF -- run: sbatch slurm/build_container.sbatch" >&2
    exit 1
fi

# Datasets must already be present: compute nodes typically have no outbound network, so a
# first-use download inside the job would fail after it has already been scheduled.
if [ ! -d "$DATA_DIR/datasets" ]; then
    echo "ERROR: $DATA_DIR/datasets missing -- run scripts/download_datasets.py on the login node" >&2
    exit 1
fi

apptainer exec --containall --nv \
    --pwd "$REPO" \
    --env USER="$USER" \
    --env CACOSE_GIT_SHA="$GIT_SHA" \
    --env XDG_CACHE_HOME="$DATA_DIR/.cache" \
    --env TORCH_HOME="$DATA_DIR/.cache/torch" \
    --env CACOSE_DATA_ROOT="$DATA_DIR/datasets" \
    --env CACOSE_OUT="$DATA_DIR" \
    --env PYTHONPATH="$REPO" \
    --bind /tmp:/tmp \
    --bind "$REPO:$REPO" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" $PYTHON -m scripts.run_single_experiment --config "$CONFIG" --seed "$SEED"

echo "[$(date)] seed $SEED finished"
