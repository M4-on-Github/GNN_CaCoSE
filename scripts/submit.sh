#!/bin/bash
# Submit a held seed sweep and print the command that releases it.
#
#   scripts/submit.sh configs/cora.yaml
#
# Submitting held means nothing starts until you look at the queue and decide. The array's %3
# cap then admits at most three tasks at a time even after a blanket release.

set -euo pipefail

CONFIG=${1:?usage: scripts/submit.sh <config.yaml> [extra sbatch args...]}
shift || true

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: no such config: ${CONFIG}" >&2
    exit 1
fi

CACOSE_ROOT=/data/${USER}/CaCoSE
mkdir -p "${CACOSE_ROOT}/logs"  # sbatch fails outright if the --output directory is missing

JOBID=$(sbatch --hold --parsable "$@" slurm/run_seeds.sbatch "${CONFIG}")

cat <<EOF

submitted held : job ${JOBID}   (${CONFIG})
logs           : ${CACOSE_ROOT}/logs/cacose_${JOBID}_*.out
results        : ${CACOSE_ROOT}/results/

Every task sits in JobHeldUser until you release it. Check first:

    squeue -j ${JOBID}

Release all 10 seeds (the array's %3 cap admits 3 at a time):

    scontrol release ${JOBID}

Or release a couple of seeds to sanity-check before committing the rest:

    scontrol release ${JOBID}_0
    scontrol release ${JOBID}_1

Cancel:

    scancel ${JOBID}
EOF
