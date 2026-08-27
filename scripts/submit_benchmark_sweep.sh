#!/bin/bash
# Submit one benchmark: 10 seeds of one config, rebuilding the container first if it is stale.
#
#   scripts/submit_benchmark_sweep.sh configs/cora.yaml
#   scripts/submit_benchmark_sweep.sh configs/mutag.yaml --after afterany:12345
#
# Three things happen here:
#
#   1. The log directory is created before sbatch opens a log file. sbatch fails outright if
#      --output points somewhere that does not exist.
#   2. scripts/ensure_container.sh compares cacose.def against the hash recorded beside the .sif
#      and submits a build job if they differ, so the sweep is chained behind it with
#      --dependency=afterok. This is what stops a stale image from silently producing results
#      under the wrong dependency versions.
#   3. The sweep is submitted and runs unattended. The array's %3 (JobArrayTaskLimit) caps it
#      at three seeds at a time; SLURM starts the next as each finishes.
#
# The caller may skip step 2 by exporting CACOSE_BUILD_JOB (a job id, or empty to mean "already
# checked, no build needed"). submit_all_benchmarks.sh does that so three chained sweeps share
# one build instead of queueing one each.

set -euo pipefail

CONFIG=${1:?usage: scripts/submit_benchmark_sweep.sh <config.yaml> [--parsable] [--after <dep>] [sbatch args...]}
shift || true

# --parsable prints only the array job id, so submit_all_benchmarks.sh can chain on it. Consumed
# here rather than forwarded, since sbatch is already given --parsable internally.
#
# --after collects extra dependency terms. They are merged into the single --dependency flag
# built below rather than passed through: sbatch keeps only the last --dependency it is given,
# so a forwarded one would silently drop the build dependency and let the sweep start against a
# half-written image.
PARSABLE=false
EXTRA_DEPS=()
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parsable) PARSABLE=true; shift ;;
        --after) EXTRA_DEPS+=("${2:?--after needs a dependency spec, e.g. afterany:12345}"); shift 2 ;;
        --after=*) EXTRA_DEPS+=("${1#*=}"); shift ;;
        --dependency|--dependency=*)
            echo "ERROR: pass dependencies with --after <spec>, not --dependency" >&2
            echo "       they must be merged with the container build dependency, not replace it" >&2
            exit 1 ;;
        *) ARGS+=("$1"); shift ;;
    esac
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

mkdir -p "$LOG_DIR" "$CACOSE_ROOT/containers"

# ── container freshness ──────────────────────────────────────────────────────
# Status goes to stderr, never stdout. Under --parsable the caller reads stdout to get the job
# id and chain the next sweep onto it; a stray status line there is captured as part of the id
# and the next --dependency is rejected with "Job dependency problem". Stderr still shows on a
# terminal, so nothing is hidden from a human.
say() { echo "$@" >&2; }

if [[ -n "${CACOSE_BUILD_JOB+set}" ]]; then
    BUILD_JOB="$CACOSE_BUILD_JOB"      # the caller already ran ensure_container.sh
else
    BUILD_JOB=$(scripts/ensure_container.sh)
fi

DEPS=("${EXTRA_DEPS[@]+"${EXTRA_DEPS[@]}"}")
[[ -n "$BUILD_JOB" ]] && DEPS+=("afterok:${BUILD_JOB}")

DEPENDENCY=""
if [[ ${#DEPS[@]} -gt 0 ]]; then
    # Comma-separated terms are ANDed by SLURM: every one must be satisfied.
    # kill-on-invalid-dep: if the build fails, cancel this sweep instead of leaving it parked in
    # DependencyNeverSatisfied, where it clogs the queue and blocks anything chained behind it.
    joined=$(IFS=,; echo "${DEPS[*]}")
    DEPENDENCY="--dependency=${joined} --kill-on-invalid-dep=yes"
    say "[sweep] $(basename "$CONFIG") waits on ${joined}"
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

if [[ -n "$BUILD_JOB" ]]; then
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
