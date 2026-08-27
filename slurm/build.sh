#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build cacose.sif from slurm/cacose.def.
#
#   sbatch slurm/build.sh        # as a batch job
#   bash   slurm/build.sh        # or directly
#
# Run from inside the repo. Self-contained: needs no other file here.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH -J cacose-build
#SBATCH --output=/data/%u/CaCoSE/logs/cacose-build_%j.out
#SBATCH --error=/data/%u/CaCoSE/logs/cacose-build_%j.err

set -e
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

DATA_DIR="/data/$USER/CaCoSE"
SIF="$DATA_DIR/containers/cacose.sif"
DEF="$REPO/slurm/cacose.def"
DEF_HASH=$(sha256sum "$DEF" | cut -d' ' -f1)

# The base image keeps its interpreter here. Addressed absolutely rather than as `python`:
# apptainer exec does not inherit the Docker image's ENV PATH, so a bare `python` is not
# guaranteed to resolve even though it works during %post and %test.
PYTHON=/opt/conda/bin/python3

echo "=========================================="
echo " Build job   : ${SLURM_JOB_ID:-local}"
echo " Node        : $(hostname)"
echo " Started     : $(date)"
echo " Target SIF  : $SIF"
echo " DEF hash    : $DEF_HASH"
echo "=========================================="

mkdir -p "$DATA_DIR/containers" "$DATA_DIR/logs" "$DATA_DIR/datasets"

# Build scratch on node-local disk: /data is mounted nodev, which apptainer warns about.
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/$USER-apptainer-$$"
export APPTAINER_CACHEDIR="$DATA_DIR/.cache/apptainer"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT

rm -f "$SIF"
echo "[$(date)] Running: apptainer build --fakeroot $SIF $DEF"
apptainer build --fakeroot "$SIF" "$DEF"
echo "$DEF_HASH" > "$SIF.def.sha256"
echo "[$(date)] Container ready: $SIF"

echo
echo "[$(date)] Verifying the stack ..."
apptainer exec --containall \
    --pwd "$REPO" \
    --env USER="$USER" \
    --env XDG_CACHE_HOME="$DATA_DIR/.cache" \
    --env TORCH_HOME="$DATA_DIR/.cache/torch" \
    --bind /tmp:/tmp \
    --bind "$REPO:$REPO" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" $PYTHON -c "
import numpy, scipy, torch, torch_geometric
print(' torch', torch.__version__, '| pyg', torch_geometric.__version__,
      '| numpy', numpy.__version__, '| scipy', scipy.__version__)
assert numpy.__version__.startswith('1.'), 'numpy must be 1.x'
assert torch.arange(3.0).numpy().sum() == 3.0, 'torch<->numpy interop broken'
print(' stack OK')
"

echo "=========================================="
echo " Container ready : $SIF"
echo " Interpreter     : $PYTHON"
echo " Finished        : $(date)"
echo "=========================================="
