#!/bin/bash
# Submit a held seed sweep, rebuilding the container first if it is stale.
#
#   scripts/submit.sh configs/cora.yaml
#
# Three things happen here:
#
#   1. The log directory is created before sbatch opens a log file. sbatch fails outright if
#      --output points somewhere that does not exist.
#   2. The container definition is hashed and compared against the hash recorded beside the
#      .sif. If they differ, a build job is submitted and the sweep is chained behind it with
#      --dependency=afterok, so tasks can only start after a successful build. This is what
#      stops a stale image from silently producing results under the wrong dependency versions.
#   3. The sweep is submitted --hold. Nothing runs until you release it; the array's %3 cap
#      then admits at most three seeds at a time.

set -euo pipefail

CONFIG=${1:?usage: scripts/submit.sh <config.yaml> [extra sbatch args...]}
shift || true

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(realpath "$0")")/.." && pwd)}"
cd "$REPO"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: no such config: $CONFIG" >&2
    exit 1
fi

CACOSE_ROOT="/data/$USER/CaCoSE"
LOG_DIR="$CACOSE_ROOT/logs"
SIF="$CACOSE_ROOT/containers/cacose.sif"
DEF="$REPO/slurm/cacose.def"
HASH_FILE="$SIF.def.sha256"

mkdir -p "$LOG_DIR" "$CACOSE_ROOT/containers"

# ── container freshness ──────────────────────────────────────────────────────
DEF_HASH=$(sha256sum "$DEF" | cut -d' ' -f1)
DEPENDENCY=""

if [[ -f "$SIF" && -f "$HASH_FILE" && "$DEF_HASH" == "$(cat "$HASH_FILE")" ]]; then
    echo "[container] up to date (${DEF_HASH:0:12})"
else
    if [[ -f "$SIF" ]]; then
        echo "[container] STALE -- cacose.def changed since ${SIF##*/} was built"
    else
        echo "[container] missing"
    fi
    BUILD_JOB=$(sbatch --parsable \
        --output="$LOG_DIR/cacose-build_%j.out" \
        --error="$LOG_DIR/cacose-build_%j.err" \
        slurm/build_container.sbatch)
    echo "[container] build job $BUILD_JOB submitted; the sweep will wait for it"
    DEPENDENCY="--dependency=afterok:${BUILD_JOB}"
fi

# ── the sweep itself ─────────────────────────────────────────────────────────
JOBID=$(sbatch --hold --parsable \
    ${DEPENDENCY:+$DEPENDENCY} \
    --output="$LOG_DIR/cacose_%A_%a.out" \
    --error="$LOG_DIR/cacose_%A_%a.err" \
    "$@" \
    slurm/run_seeds.sbatch "$CONFIG")

cat <<EOF

submitted held : job ${JOBID}   (${CONFIG})
logs           : ${LOG_DIR}/cacose_${JOBID}_*.out
results        : ${CACOSE_ROOT}/results/
EOF

if [[ -n "$DEPENDENCY" ]]; then
    cat <<EOF
depends on     : build job ${BUILD_JOB} (afterok)

The sweep is BOTH held and waiting on the build. Watch the build first:
    tail -f ${LOG_DIR}/cacose-build_${BUILD_JOB}.out
EOF
fi

cat <<EOF

Every task sits in JobHeldUser until you release it. Check first:

    squeue -j ${JOBID}

Release all 10 seeds (the array's %3 cap admits 3 at a time):

    scontrol release ${JOBID}

Or release one seed to sanity-check before committing the rest:

    scontrol release ${JOBID}_0

Cancel:

    scancel ${JOBID}
EOF
