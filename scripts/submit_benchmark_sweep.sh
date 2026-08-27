#!/bin/bash
# Submit a held seed sweep, rebuilding the container first if it is stale.
#
#   scripts/submit_benchmark_sweep.sh configs/cora.yaml
#
# Three things happen here:
#
#   1. The log directory is created before sbatch opens a log file. sbatch fails outright if
#      --output points somewhere that does not exist.
#   2. The container definition is hashed and compared against the hash recorded beside the
#      .sif. If they differ, a build job is submitted and the sweep is chained behind it with
#      --dependency=afterok, so tasks can only start after a successful build. This is what
#      stops a stale image from silently producing results under the wrong dependency versions.
#   3. The sweep is submitted and runs unattended. The array's %3 (JobArrayTaskLimit) caps it
#      at three seeds at a time; SLURM starts the next as each finishes.

set -euo pipefail

CONFIG=${1:?usage: scripts/submit_benchmark_sweep.sh <config.yaml> [--parsable] [extra sbatch args...]}
shift || true

# --parsable prints only the array job id, so submit_all_benchmarks.sh can chain on it. Consumed here
# rather than forwarded, since sbatch is already given --parsable internally.
PARSABLE=false
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--parsable" ]]; then PARSABLE=true; else ARGS+=("$arg"); fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

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
    [[ "$PARSABLE" == true ]] || echo "[container] up to date (${DEF_HASH:0:12})"
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
# Not held: the array's %3 (JobArrayTaskLimit) is the only throttle. SLURM starts three tasks
# and begins the next as each finishes, so concurrency is capped without a manual gate.
JOBID=$(sbatch --parsable \
    ${DEPENDENCY:+$DEPENDENCY} \
    --output="$LOG_DIR/cacose_%A_%a.out" \
    --error="$LOG_DIR/cacose_%A_%a.err" \
    "$@" \
    slurm/train_benchmark_array.sbatch "$CONFIG")

if [[ "$PARSABLE" == true ]]; then
    echo "$JOBID"
    exit 0
fi

cat <<EOF

submitted : job ${JOBID}   (${CONFIG})
logs      : ${LOG_DIR}/cacose_${JOBID}_*.out
results   : ${CACOSE_ROOT}/results/
EOF

if [[ -n "$DEPENDENCY" ]]; then
    cat <<EOF
waiting on: build job ${BUILD_JOB} (afterok)

Tasks stay pending with REASON=Dependency until the build succeeds:
    tail -f ${LOG_DIR}/cacose-build_${BUILD_JOB}.out
EOF
fi

cat <<EOF

Runs on its own, 3 seeds at a time (REASON=JobArrayTaskLimit for the queued ones).

    squeue -j ${JOBID}
    tail -f ${LOG_DIR}/cacose_${JOBID}_0.out

Cancel:

    scancel ${JOBID}
EOF
