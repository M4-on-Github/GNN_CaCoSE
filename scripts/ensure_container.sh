#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Make sure cacose.sif matches cacose.def, submitting one build job if it does not.
#
#   BUILD=$(scripts/ensure_container.sh)     # empty if already up to date
#
# Prints the build job id to stdout, or nothing at all when no build is needed. Every status
# line goes to stderr, so the caller can capture stdout as a bare job id.
#
# This lives in its own file because the check has exactly one correct number of callers per
# submission: one. submit_benchmark_sweep.sh used to run it inline, so submit_all_benchmarks.sh
# -- which calls that script once per dataset -- submitted three identical build jobs, none of
# which had run yet when the next check ran, so each one saw a missing .sif and queued another.
# Three concurrent `apptainer build --force` calls write the same path.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(realpath "$0")")/.." && pwd)}"
cd "$REPO"

CACOSE_ROOT="/data/$USER/CaCoSE"
LOG_DIR="$CACOSE_ROOT/logs"
SIF="$CACOSE_ROOT/containers/cacose.sif"
DEF="$REPO/slurm/cacose.def"
HASH_FILE="$SIF.def.sha256"

mkdir -p "$LOG_DIR" "$CACOSE_ROOT/containers"

say() { echo "$@" >&2; }

DEF_HASH=$(sha256sum "$DEF" | cut -d' ' -f1)

# The sidecar is written by the build job only after the image has been exec'd successfully, so
# a matching hash means "this image was verified", not merely "this image exists".
if [[ -f "$SIF" && -f "$HASH_FILE" && "$DEF_HASH" == "$(cat "$HASH_FILE")" ]]; then
    say "[container] up to date (${DEF_HASH:0:12})"
    exit 0
fi

if [[ -f "$SIF" ]]; then
    say "[container] STALE or unverified -- rebuilding ${SIF##*/}"
else
    say "[container] missing -- building it first"
fi

BUILD_JOB=$(sbatch --parsable \
    --output="$LOG_DIR/cacose-build_%j.out" \
    --error="$LOG_DIR/cacose-build_%j.err" \
    slurm/build_container.sbatch)

say "[container] build job $BUILD_JOB submitted; sweeps will wait for it"
echo "$BUILD_JOB"
