#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build cacose.sif from slurm/cacose.def.
#
#   sbatch slurm/build_container.sbatch        # as a batch job
#   bash   slurm/build_container.sbatch        # or directly
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

# Build scratch goes on the NAS, NOT /tmp.
#
# /tmp on these nodes is tmpfs, i.e. RAM. mksquashfs unpacks the entire image there before
# compressing it, so pointing APPTAINER_TMPDIR at /tmp charges several gigabytes against the
# job's memory cgroup. When it runs out the build does not fail loudly: it writes a .sif whose
# header is valid -- `apptainer inspect` succeeds -- but whose squashfs payload is truncated, and
# every later `apptainer exec` dies with "input/output error", including exec of /bin/echo.
#
# That cost a full round of failed benchmarks. Apptainer warns that /data is mounted nodev during
# the build; that warning is cosmetic and the build works anyway.
export APPTAINER_TMPDIR="$DATA_DIR/.cache/apptainer_tmp"
export APPTAINER_CACHEDIR="$DATA_DIR/.cache/apptainer"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# --force replaces an existing image in one step. Deleting it first leaves nothing behind if the
# build then fails.
echo "[$(date)] Running: apptainer build --force --fakeroot $SIF $DEF"
apptainer build --force --fakeroot "$SIF" "$DEF"
echo "$DEF_HASH" > "$SIF.def.sha256"
echo "[$(date)] Container ready: $SIF"

echo
echo "[$(date)] Verifying the stack ..."

# Retry, and never fail the job on this check. An "input/output error" here means the image is
# unusable -- see the APPTAINER_TMPDIR comment above for the cause that produced one -- but the
# right response is a loud warning, not aborting: a failed build job leaves every benchmark
# chained behind it stuck in DependencyNeverSatisfied.
sync || true

verify() {
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
}

VERIFIED=false
for attempt in 1 2 3 4 5; do
    if verify; then
        VERIFIED=true
        break
    fi
    echo "[$(date)] verification attempt ${attempt} failed"
    sleep 10
done

if [ "$VERIFIED" != true ]; then
    # Warn rather than exit: a failed build job leaves every benchmark chained behind it stuck
    # in DependencyNeverSatisfied. But do not pretend the image is fine -- a build whose %test
    # passed can still produce an unusable .sif, which is exactly what a truncated squashfs
    # looks like.
    echo "WARNING: the image cannot be executed. It is probably NOT usable." >&2
    echo "         A %test that passed does not clear it: %test runs against the build tree," >&2
    echo "         not the finished .sif." >&2
    echo "         'input/output error' here means a truncated squashfs -- most likely the" >&2
    echo "         build ran out of memory with APPTAINER_TMPDIR on tmpfs, or --mem was too" >&2
    echo "         low. Confirm with:" >&2
    echo "           apptainer inspect $SIF        # header only; succeeds even when truncated" >&2
    echo "           apptainer exec $SIF /bin/echo hi" >&2
    echo "         Then rebuild with more memory." >&2
fi

echo "=========================================="
echo " Container ready : $SIF"
echo " Interpreter     : $PYTHON"
echo " Finished        : $(date)"
echo "=========================================="
