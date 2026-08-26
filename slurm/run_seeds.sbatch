#!/bin/bash
# Run one config across 10 seeds as a SLURM array.
#
#   sbatch --hold slurm/run_seeds.sbatch configs/cora.yaml
#   scontrol release <jobid>
#
# Two independent throttles, because they fail differently:
#   --hold   parks every task in JobHeldUser until you release it. A manual gate.
#   %3       caps concurrency at 3 regardless. Still holds if you release everything at once.
#
# No --nodelist: peak usage is ~1-2 GB VRAM, so any node in the partition will do and the
# queue is shorter for it. cu121 in the container covers every GPU the cluster has.

#SBATCH --job-name=cacose
#SBATCH --partition=pleiades
#SBATCH --chdir=/home/mmyatmau/CaCoSE
#SBATCH --array=0-9%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/mmyatmau/CaCoSE/logs/%x_%A_%a.out
#SBATCH --error=/data/mmyatmau/CaCoSE/logs/%x_%A_%a.err

set -euo pipefail

CONFIG=${1:?usage: sbatch --hold slurm/run_seeds.sbatch <config.yaml>}

# Resolve apptainer/singularity exactly as the build job does. Without this, a cluster where the
# runtime sits behind an environment module would fail every array task with "command not found".
source "$(dirname "${BASH_SOURCE[0]}")/_runtime.sh"
require_container_runtime

CACOSE_HOME=/home/${USER}/CaCoSE
CACOSE_ROOT=/data/${USER}/CaCoSE
SIF=${CACOSE_ROOT}/containers/cacose.sif

# Code from /home, everything written to /data.
export CACOSE_DATA_ROOT=${CACOSE_ROOT}/datasets
export CACOSE_OUT=${CACOSE_ROOT}
export PYTHONPATH=${CACOSE_HOME}

echo "host      : $(hostname)"
echo "gpu       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "config    : ${CONFIG}"
echo "seed      : ${SLURM_ARRAY_TASK_ID}"
echo "code      : ${CACOSE_HOME}"
echo "output    : ${CACOSE_OUT}"

if [[ ! -f "${SIF}" ]]; then
    echo "ERROR: container not found at ${SIF} -- build it first (see RUNBOOK.md)" >&2
    exit 1
fi

# Datasets must already be present: compute nodes typically have no outbound network, so a
# first-use download inside the job would fail. scripts/prefetch_data.py runs on the login node.
if [[ ! -d "${CACOSE_DATA_ROOT}" ]]; then
    echo "ERROR: ${CACOSE_DATA_ROOT} missing -- run scripts/prefetch_data.py on head1 first" >&2
    exit 1
fi

"${CONTAINER_CMD}" exec --nv \
    --bind "${CACOSE_ROOT}:${CACOSE_ROOT}" \
    --bind "${CACOSE_HOME}:${CACOSE_HOME}" \
    "${SIF}" \
    python -m scripts.run \
        --config "${CONFIG}" \
        --seed "${SLURM_ARRAY_TASK_ID}"

echo "task ${SLURM_ARRAY_TASK_ID} finished"
