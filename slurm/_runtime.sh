#!/bin/bash
# Locate the container runtime. Sourced by build_container.sbatch and run_seeds.sbatch so both
# resolve it the same way.
#
# `apptainer: command not found` on a compute node has three common causes: the tool is still
# installed under its former name (singularity), it lives behind an environment module, or it is
# only on the login node. This tries each in turn and sets $CONTAINER_CMD.

detect_container_runtime() {
    # 1. Already on PATH?
    for candidate in apptainer singularity; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            CONTAINER_CMD=$(command -v "${candidate}")
            return 0
        fi
    done

    # 2. Behind an environment module? `module` is a shell function, so it may need sourcing
    #    before it exists in a non-interactive batch shell.
    if ! command -v module >/dev/null 2>&1; then
        for init in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
            [[ -f "${init}" ]] && source "${init}" && break
        done
    fi
    if command -v module >/dev/null 2>&1; then
        for mod in apptainer singularity; do
            if module load "${mod}" >/dev/null 2>&1; then
                for candidate in apptainer singularity; do
                    if command -v "${candidate}" >/dev/null 2>&1; then
                        CONTAINER_CMD=$(command -v "${candidate}")
                        echo "note: loaded module '${mod}'"
                        return 0
                    fi
                done
            fi
        done
    fi

    # 3. Installed but outside PATH?
    for path in /usr/local/bin /opt/apptainer/bin /opt/singularity/bin /usr/bin; do
        for candidate in apptainer singularity; do
            if [[ -x "${path}/${candidate}" ]]; then
                CONTAINER_CMD="${path}/${candidate}"
                return 0
            fi
        done
    done

    return 1
}

require_container_python() {
    # Find the interpreter *inside* an image and set $CONTAINER_PYTHON.
    #
    # `apptainer exec <sif> python` fails with "executable file not found in $PATH" when the
    # image relies on Docker's ENV PATH: Apptainer honours that during %post and %test but not
    # at exec time. The definition now exports PATH explicitly, so a freshly built image works
    # with a bare `python` -- this probe keeps images built before that fix usable too, and
    # covers base images that put Python somewhere else entirely.
    local sif=$1
    local candidate
    for candidate in python python3 /opt/conda/bin/python /usr/local/bin/python /usr/bin/python3; do
        if "${CONTAINER_CMD}" exec "${sif}" "${candidate}" -c pass >/dev/null 2>&1; then
            CONTAINER_PYTHON=${candidate}
            echo "container python : ${CONTAINER_PYTHON}"
            return 0
        fi
    done
    echo "ERROR: no usable python found inside ${sif}" >&2
    echo "       tried: python python3 /opt/conda/bin/python /usr/local/bin/python /usr/bin/python3" >&2
    echo "       inspect with: ${CONTAINER_CMD} exec ${sif} sh -c 'ls /opt/conda/bin | head'" >&2
    return 1
}

require_container_runtime() {
    if ! detect_container_runtime; then
        cat >&2 <<'MSG'
ERROR: no container runtime found (looked for apptainer and singularity).

Tried: PATH, environment modules, and the usual install locations.

Diagnose with:
    command -v apptainer singularity
    module avail 2>&1 | grep -iE 'apptainer|singularity'
    ls -1 /usr/local/bin /opt 2>/dev/null | grep -iE 'apptainer|singularity'

If it exists only on the login node, the container has to be built there.
MSG
        return 1
    fi
    echo "container runtime: ${CONTAINER_CMD} ($(${CONTAINER_CMD} --version 2>&1 | head -1))"
}
