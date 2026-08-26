#!/bin/bash
# Build the CaCoSE container as a batch job.
#
#   sbatch slurm/build_container.sbatch
#
# Batch rather than an interactive srun so the build survives a dropped SSH connection and
# leaves a log you can read afterwards.
#
# No GPU is requested: a build unpacks a base image and runs pip, which is I/O bound. Holding a
# GPU for 10 minutes to do that would take a scarce resource for nothing. The consequence is
# that the definition's %test section prints "cuda available: False" -- expected here, and not
# a problem. CUDA is exercised for real by the first training job.

#SBATCH --job-name=cacose-build
#SBATCH --partition=pleiades
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/%u/CaCoSE/logs/%x_%j.out
#SBATCH --error=/data/%u/CaCoSE/logs/%x_%j.err

set -euo pipefail

# SLURM starts a job in the directory sbatch was run from, so derive the repo rather than
# hardcoding a path: the clone can live anywhere and be named anything.
CACOSE_HOME=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "${CACOSE_HOME}"

CACOSE_ROOT=/data/${USER}/CaCoSE
SIF=${CACOSE_ROOT}/containers/cacose.sif
DEF=${CACOSE_HOME}/slurm/cacose.def

if [[ ! -f "${DEF}" ]]; then
    echo "ERROR: ${DEF} not found -- submit from inside the CaCoSE clone:" >&2
    echo "       cd <repo> && sbatch slurm/build_container.sbatch" >&2
    exit 1
fi

# Resolve helpers from the repo, NOT from the script's own path. SLURM copies the batch script
# to /var/spool/slurmd/job<id>/slurm_script before running it, so $BASH_SOURCE points at that
# copy and its siblings do not exist.
RUNTIME_SH=${CACOSE_HOME}/slurm/_runtime.sh
if [[ ! -f "${RUNTIME_SH}" ]]; then
    echo "ERROR: ${RUNTIME_SH} not found -- submit from inside the CaCoSE clone" >&2
    exit 1
fi
source "${RUNTIME_SH}"
require_container_runtime

echo "host   : $(hostname)"
echo "def    : ${DEF}"
echo "target : ${SIF}"

# Keep the image cache and build scratch off /home. Pulling and unpacking the CUDA base image
# writes several GB, and /home is the 500GB quota'd partition.
export APPTAINER_CACHEDIR=${CACOSE_ROOT}/.cache/apptainer
export SINGULARITY_CACHEDIR=${APPTAINER_CACHEDIR}
export APPTAINER_TMPDIR=${CACOSE_ROOT}/.cache/tmp
export SINGULARITY_TMPDIR=${APPTAINER_TMPDIR}
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${CACOSE_ROOT}/containers"

if [[ -f "${SIF}" ]]; then
    echo "note   : ${SIF} already exists; moving it aside rather than overwriting"
    mv "${SIF}" "${SIF}.$(date +%Y%m%d-%H%M%S).bak"
fi

echo "building (5-15 min, mostly pulling the base image)..."

# Unprivileged builds work on most modern setups via user namespaces. Where they do not,
# --fakeroot is the usual remedy, so try it before giving up.
if ! "${CONTAINER_CMD}" build "${SIF}" "${DEF}"; then
    echo "plain build failed; retrying with --fakeroot" >&2
    "${CONTAINER_CMD}" build --fakeroot "${SIF}" "${DEF}"
fi

echo
echo "built: ${SIF}"
ls -lh "${SIF}"

echo
echo "verifying the image can import the stack..."
"${CONTAINER_CMD}" exec "${SIF}" python -c "
import numpy, scipy, torch, torch_geometric
print('  torch', torch.__version__, '| pyg', torch_geometric.__version__,
      '| numpy', numpy.__version__, '| scipy', scipy.__version__)
assert numpy.__version__.startswith('1.'), 'numpy must be 1.x'
assert torch.arange(3.0).numpy().sum() == 3.0, 'torch<->numpy interop broken'
print('  stack OK')
"

echo
echo "next: prefetch datasets on the login node, then submit a sweep"
echo "  ${CONTAINER_CMD} exec ${SIF} python -m scripts.prefetch_data --all"
echo "  scripts/submit.sh configs/cora.yaml"
