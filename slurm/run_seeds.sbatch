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
#SBATCH --array=0-9%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/%u/CaCoSE/logs/%x_%A_%a.out
#SBATCH --error=/data/%u/CaCoSE/logs/%x_%A_%a.err

set -euo pipefail

CONFIG=${1:?usage: sbatch --hold slurm/run_seeds.sbatch <config.yaml>}

# The repo is wherever sbatch was run from -- SLURM starts a job in the submission directory.
# Deriving it rather than hardcoding /home/<user>/CaCoSE means the clone can live anywhere and
# be named anything. This must come first: everything below is resolved relative to it.
CACOSE_HOME=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "${CACOSE_HOME}"

CACOSE_ROOT=/data/${USER}/CaCoSE
SIF=${CACOSE_ROOT}/containers/cacose.sif

if [[ ! -f "${CACOSE_HOME}/pyproject.toml" ]]; then
    echo "ERROR: ${CACOSE_HOME} does not look like the CaCoSE repo (no pyproject.toml)." >&2
    echo "       Submit from inside the clone: cd <repo> && sbatch slurm/run_seeds.sbatch ..." >&2
    exit 1
fi

# Resolve apptainer/singularity the same way the build job does. Source the helper from the repo,
# NOT from the script's own path: SLURM copies the batch script to /var/spool/slurmd/job<id>/
# before running it, so $BASH_SOURCE points at that copy and its siblings do not exist.
RUNTIME_SH=${CACOSE_HOME}/slurm/_runtime.sh
if [[ ! -f "${RUNTIME_SH}" ]]; then
    echo "ERROR: ${RUNTIME_SH} not found -- submit from inside the CaCoSE clone" >&2
    exit 1
fi
source "${RUNTIME_SH}"
require_container_runtime

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
