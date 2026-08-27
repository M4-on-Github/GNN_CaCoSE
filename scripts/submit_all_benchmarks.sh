#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Run all three reproduction sweeps, one dataset at a time.
#
#   scripts/submit_all_benchmarks.sh
#   scripts/submit_all_benchmarks.sh configs/cora.yaml configs/mutag.yaml    # or a subset
#
# Each sweep is 10 seeds capped at 3 concurrent by the array's %3. Submitting all three at once
# would allow 9 concurrent, since each array caps independently -- so each dataset is chained
# behind the previous with afterany. Never more than 3 of your tasks run at a time.
#
# afterany, not afterok: a dataset that fails should not silently block the ones after it. Check
# the logs rather than assuming a quiet queue means success.
#
# The container is checked ONCE here, and every sweep is chained behind that single build with
# afterok. Letting each sweep run its own check submitted three build jobs: the first build has
# not run by the time the second check happens, so it still sees a missing .sif. Each sweep
# after the first therefore carries two dependency terms -- the shared build, and the previous
# dataset -- which is why they go through --after and are merged into one --dependency flag.
#
# Cora and MUTAG take about a minute each; Chameleon about 13, so it goes last.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
cd "$REPO"

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(configs/cora.yaml configs/mutag.yaml configs/chameleon.yaml)
fi

for cfg in "${CONFIGS[@]}"; do
    if [ ! -f "$cfg" ]; then
        echo "ERROR: no such config: $cfg" >&2
        exit 1
    fi
done

CACOSE_ROOT="/data/$USER/CaCoSE"

# One build for the whole run. Exported even when empty: that is how submit_benchmark_sweep.sh
# knows the check has already been made and must not repeat it.
CACOSE_BUILD_JOB=$(scripts/ensure_container.sh)
export CACOSE_BUILD_JOB

PREV=""
IDS=()

for cfg in "${CONFIGS[@]}"; do
    if [ -n "$PREV" ]; then
        JOB=$(scripts/submit_benchmark_sweep.sh "$cfg" --parsable --after "afterany:${PREV}")
        WHEN="after $PREV"
    else
        JOB=$(scripts/submit_benchmark_sweep.sh "$cfg" --parsable)
        WHEN="starts now"
    fi

    # A job id is all that may reach stdout. Anything else means a status line leaked into it,
    # and chaining --dependency on that produces only SLURM's opaque "Job dependency problem".
    if ! [[ "$JOB" =~ ^[0-9]+$ ]]; then
        echo "ERROR: expected a job id from submit_benchmark_sweep.sh, got: '$JOB'" >&2
        echo "       status output must go to stderr so stdout carries only the id" >&2
        exit 1
    fi
    echo "queued : $(basename "$cfg" .yaml)  job $JOB   ($WHEN)"
    IDS+=("$JOB")
    PREV="$JOB"
done

if [ -n "$CACOSE_BUILD_JOB" ]; then
    echo "build  : job $CACOSE_BUILD_JOB   (all ${#CONFIGS[@]} sweeps wait on it)"
fi

cat <<EOF

All ${#CONFIGS[@]} sweeps queued, running one dataset at a time (3 seeds concurrent within each).

    squeue -u $USER
    tail -f ${CACOSE_ROOT}/logs/cacose_${IDS[0]}_0.out

When everything finishes:

    CACOSE_OUT=${CACOSE_ROOT} python3 -m scripts.aggregate_benchmark_results
    tar czf cacose_results.tgz -C ${CACOSE_ROOT} results

Cancel everything: scancel ${IDS[*]}${CACOSE_BUILD_JOB:+ $CACOSE_BUILD_JOB}
EOF
