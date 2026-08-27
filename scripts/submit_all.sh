#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Run all three reproduction sweeps, one dataset at a time.
#
#   scripts/submit_all.sh
#   scripts/submit_all.sh configs/cora.yaml configs/mutag.yaml    # or a subset
#
# Each sweep is 10 seeds capped at 3 concurrent by the array's %3. Submitting all three at once
# would allow 9 concurrent, since each array caps independently -- so each dataset is chained
# behind the previous with --dependency=afterany. Never more than 3 of your tasks run at a time.
#
# afterany, not afterok: a dataset that fails should not silently block the ones after it. Check
# the logs rather than assuming a quiet queue means success.
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

CACOSE_ROOT="/data/$USER/CaCoSE"
PREV=""
IDS=()

for cfg in "${CONFIGS[@]}"; do
    if [ ! -f "$cfg" ]; then
        echo "ERROR: no such config: $cfg" >&2
        exit 1
    fi
    if [ -n "$PREV" ]; then
        JOB=$(scripts/submit.sh "$cfg" --parsable --dependency="afterany:${PREV}")
        echo "queued : $(basename "$cfg" .yaml)  job $JOB   (after $PREV)"
    else
        JOB=$(scripts/submit.sh "$cfg" --parsable)
        echo "queued : $(basename "$cfg" .yaml)  job $JOB   (starts now)"
    fi
    IDS+=("$JOB")
    PREV="$JOB"
done

cat <<EOF

All ${#CONFIGS[@]} sweeps queued, running one dataset at a time (3 seeds concurrent within each).

    squeue -u $USER
    tail -f ${CACOSE_ROOT}/logs/cacose_${IDS[0]}_0.out

When everything finishes:

    CACOSE_OUT=${CACOSE_ROOT} python3 -m scripts.sweep_seeds
    tar czf cacose_results.tgz -C ${CACOSE_ROOT} results

Cancel everything: scancel ${IDS[*]}
EOF
