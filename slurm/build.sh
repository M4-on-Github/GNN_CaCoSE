#!/bin/bash
# Build the CaCoSE container. Self-contained: needs no other file in this repo.
#
#   sbatch slurm/build.sh      # as a batch job
#   bash   slurm/build.sh      # or directly, anywhere apptainer works
#
# Run it from inside the repo.

#SBATCH --job-name=cacose-build
#SBATCH --partition=pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/%u/CaCoSE/logs/cacose-build_%j.out
#SBATCH --error=/data/%u/CaCoSE/logs/cacose-build_%j.err

set -e

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

OUT=/data/$USER/CaCoSE
SIF=$OUT/containers/cacose.sif
DEF=$REPO/slurm/cacose.def

mkdir -p "$OUT/containers" "$OUT/logs" "$OUT/datasets"

# apptainer is sometimes called singularity, sometimes behind a module
APP=$(command -v apptainer || command -v singularity || true)
if [ -z "$APP" ]; then
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
    APP=$(command -v apptainer || command -v singularity || true)
fi
if [ -z "$APP" ]; then
    echo "ERROR: no apptainer or singularity found" >&2
    exit 1
fi

# node-local scratch: /data is mounted nodev, which apptainer warns about
export APPTAINER_TMPDIR=${TMPDIR:-/tmp}/$USER-apptainer-$$
export SINGULARITY_TMPDIR=$APPTAINER_TMPDIR
export APPTAINER_CACHEDIR=$OUT/.cache/apptainer
export SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT

echo "runtime : $APP"
echo "def     : $DEF"
echo "target  : $SIF"
echo

rm -f "$SIF"
"$APP" build --fakeroot "$SIF" "$DEF"

sha256sum "$DEF" | cut -d' ' -f1 > "$SIF.def.sha256"

echo
echo "built: $SIF"
ls -lh "$SIF"
echo
# Try `python` first; fall back to the interpreter's real location if PATH still hides it.
for PY in python /opt/conda/bin/python python3; do
    if "$APP" exec "$SIF" "$PY" -c pass 2>/dev/null; then
        echo "interpreter: $PY"
        "$APP" exec "$SIF" "$PY" -c "import torch, torch_geometric, numpy; print('torch', torch.__version__, '| pyg', torch_geometric.__version__, '| numpy', numpy.__version__)"
        echo "OK"
        exit 0
    fi
done
echo "ERROR: no working python inside $SIF" >&2
"$APP" exec "$SIF" sh -c 'echo PATH=$PATH; ls -l /usr/local/bin/python* 2>&1; ls /opt/conda/bin/python* 2>&1' >&2
exit 1
