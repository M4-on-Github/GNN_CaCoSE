"""Guard the spec against the code drifting away from it.

The design spec's ambiguity table is the single source of truth for what the paper leaves
underdetermined and what we chose instead. Configs and code cross-reference those rows by number
("ambiguity #4"), which is useful right up until someone renumbers the table or invents a
reference that does not exist.

An earlier version of CLAUDE.md drifted exactly this way -- it restated the CaEF rule and ended up
contradicting the spec on whether triadic support is computed inside G_k. These tests make that
class of drift fail in CI instead of surfacing as a wrong decomposition months later.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "writeups" / "phase1_implementation.tex"


def spec_text() -> str:
    if not SPEC.exists():  # pragma: no cover
        pytest.skip("design spec not present")
    return SPEC.read_text(encoding="utf-8")


def ambiguity_ids() -> set[str]:
    """Row numbers in the spec's ambiguity table, e.g. {'1', ..., '6a', '6b', '11'}."""
    rows = re.findall(r"^(\d+[ab]?)\s*&", spec_text(), flags=re.MULTILINE)
    return set(rows)


def referenced_ids() -> dict[str, list[str]]:
    """Every 'ambiguity #N' / '(#N)' reference in configs and package code."""
    found: dict[str, list[str]] = {}
    files = list((REPO / "configs").glob("*.yaml")) + list((REPO / "cacose").rglob("*.py"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"(?:ambiguity\s*#|\(#)(\d+[ab]?)", text, flags=re.IGNORECASE):
            found.setdefault(match.group(1), []).append(str(path.relative_to(REPO)))
    return found


def test_spec_ambiguity_table_is_parseable():
    ids = ambiguity_ids()
    assert len(ids) >= 10, f"expected the full ambiguity table, found {sorted(ids)}"
    assert {"1", "2", "7"} <= ids


def test_every_referenced_ambiguity_exists_in_the_spec():
    """A config commenting '(#12)' when the table stops at 11 is a silent lie."""
    known = ambiguity_ids()
    dangling = {ref: files for ref, files in referenced_ids().items() if ref not in known}
    assert not dangling, (
        f"references to ambiguities that are not in the spec table: {dangling}. "
        f"Spec has: {sorted(known)}"
    )


def test_settled_ambiguities_are_marked_resolved():
    """#2 (support inside G_k) and #7 (Chameleon split) are settled by evidence, not assumption."""
    text = spec_text()
    for row_start in ("2 & Triadic support", "7 & Chameleon"):
        idx = text.index(row_start)
        row = text[idx : text.index(chr(92) * 2, idx)]
        assert "Resolved" in row, f"row starting '{row_start}' should be marked Resolved"


def test_dropout_is_recorded_as_unspecified_by_the_paper():
    """Dropout is our choice, not the paper's -- it must be logged, or it reads as reproduction."""
    text = spec_text()
    assert re.search(r"11\s*&\s*Dropout", text), "dropout must appear in the ambiguity table"


def test_spec_records_the_generalised_cache_key():
    """The code keys the cache by decomposer id; the spec must not still claim the old form."""
    text = spec_text()
    assert "decomposer_id" in text
    assert "cache key: (dataset, delta, caef_mode)" not in text


SCRIPTS = ["run_seeds.sbatch", "build.sh"]


def script_text(name: str) -> str:
    return (REPO / "slurm" / name).read_text(encoding="utf-8")


def directives(name: str) -> str:
    lines = [ln for ln in script_text(name).splitlines() if ln.startswith("#SBATCH")]
    return "\n".join(lines)


def test_runbook_and_container_agree_on_paths():
    runbook = (REPO / "RUNBOOK.md").read_text(encoding="utf-8")
    sbatch = script_text("run_seeds.sbatch")
    for fragment in ("/data/", "containers/cacose.sif", "datasets"):
        assert fragment in runbook and fragment in sbatch, f"{fragment} missing from one of them"


def test_sweep_keeps_the_concurrency_cap_and_no_node_pin():
    """Two throttles were a deliberate decision; losing either changes cluster impact."""
    d = directives("run_seeds.sbatch")
    assert "--array=0-9%3" in d, "the %3 concurrency cap must stay"
    assert "--nodelist" not in d, "pinning a node was deliberately dropped"
    assert "-p pleiades" in d or "--partition=pleiades" in d
    assert d.count("=/data/") == 2, "both --output and --error belong under /data"


@pytest.mark.parametrize("script", SCRIPTS)
def test_sbatch_directives_carry_no_hardcoded_username(script):
    """#SBATCH lines do not expand $USER, so a literal username there breaks for anyone else.

    SLURM's %u placeholder does expand in output patterns. --chdir has no equivalent, which is
    why it was dropped: a job already starts in the submission directory.
    """
    d = directives(script)
    assert "mmyatmau" not in d, "hardcoded username in a #SBATCH directive"
    assert "--chdir" not in d, "--chdir cannot expand $USER; rely on SLURM_SUBMIT_DIR"
    assert "/data/%u/" in d, "log paths should use SLURM's %u placeholder"


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_locate_the_repo_themselves(script):
    """SLURM copies the batch script to /var/spool/slurmd/job<id>/ before running it, so paths
    must come from $SLURM_SUBMIT_DIR rather than the script's own location."""
    assert "SLURM_SUBMIT_DIR" in script_text(script)


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_use_the_absolute_interpreter_path(script):
    """`apptainer exec <sif> python` fails with "executable file not found in $PATH": the base
    image exposes Python through Docker's ENV PATH, which apptainer honours during %post and
    %test but not at exec time. Addressing the interpreter absolutely sidesteps it entirely --
    the approach the sibling ONR_CAI project uses on this same cluster."""
    text = script_text(script)
    assert "PYTHON=/opt/conda/bin/python3" in text
    offenders = [
        ln.strip()
        for ln in text.splitlines()
        if not ln.strip().startswith("#")
        and "apptainer exec" in ln
        and re.search(r"(?<![\w/${])python", ln)
    ]
    assert not offenders, f"bare `python` handed to apptainer exec: {offenders}"


def test_container_test_section_avoids_an_indented_heredoc():
    """The %test body is indented for readability, and a heredoc passes that indentation
    straight to the interpreter -- Python then rejects the script with IndentationError and the
    whole build fails at the last step. `python -c` with source at column 0 avoids it."""
    text = (REPO / "slurm" / "cacose.def").read_text(encoding="utf-8")
    section = text[text.index("%test") : text.index("%labels")]
    assert "<<" not in section, "no heredoc in %test; use python -c with column-0 source"

    marker = 'python -c "'
    assert marker in section
    code = section[section.index(marker) + len(marker) :]
    code = code[: code.index('\n"')]  # up to the closing quote on its own line

    indented = [ln for ln in code.splitlines() if ln.startswith((" ", "\t"))]
    assert not indented, f"python source inside %test must start at column 0: {indented[:2]}"
    compile(code, "cacose.def:%test", "exec")  # must at least parse


def def_section(name: str) -> str:
    """One %section of the container definition, up to the next section header.

    Split on headers at column 0 rather than on the bare name: section names also occur inside
    the comments, and slicing on those truncates the section being examined.
    """
    text = (REPO / "slurm" / "cacose.def").read_text(encoding="utf-8")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"%{name}"))
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("%")), len(lines)
    )
    return "\n".join(lines[start:end])


def test_container_definition_exports_path_for_exec():
    """%environment must put the interpreter on PATH, or a rebuilt image repeats the failure."""
    env = def_section("environment")
    assert "/opt/conda/bin" in env, "PATH must include the base image's conda bin"
    assert "export PATH" in env


def test_container_and_pyproject_pin_the_same_versions():
    """The mirrored venv only means something while these two agree."""
    proj = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    dfn = (REPO / "slurm" / "cacose.def").read_text(encoding="utf-8")
    for pin in ("2.5.1", "2.6.1", "numpy<2"):
        assert pin in proj, f"{pin} missing from pyproject.toml"
        assert pin in dfn, f"{pin} missing from cacose.def"


def test_container_post_symlinks_python():
    """A symlink in /usr/local/bin (on Apptainer's default PATH) makes `python` resolve at exec
    time even when the base image only provides it through Docker's ENV PATH."""
    post = def_section("post")
    assert "/usr/local/bin/python" in post and "ln -sf" in post


def test_submit_detects_a_stale_container_and_chains_the_rebuild():
    """A .sif built from an older definition would silently run the wrong dependency versions.

    submit.sh hashes cacose.def, compares it against the hash recorded beside the image, and
    chains the sweep behind a rebuild with --dependency=afterok when they differ.
    """
    submit = (REPO / "scripts" / "submit.sh").read_text(encoding="utf-8")
    build = (REPO / "slurm" / "build.sh").read_text(encoding="utf-8")

    assert "sha256sum" in submit and "def.sha256" in submit
    assert "--dependency=afterok" in submit
    assert "--hold" not in submit, "sweeps run unattended; %3 is the only throttle"
    # the build must write the sidecar, or submit.sh would rebuild on every invocation
    assert "def.sha256" in build and "sha256sum" in build


def test_submit_creates_the_log_dir_before_sbatch_opens_it():
    """sbatch fails outright if --output names a directory that does not exist.

    Compares line numbers of real commands: the word "sbatch" also appears in the comments
    explaining this very requirement, so a plain substring search finds prose, not a call.
    """
    lines = (REPO / "scripts" / "submit.sh").read_text(encoding="utf-8").splitlines()
    code = [(i, ln) for i, ln in enumerate(lines) if not ln.strip().startswith("#")]
    mkdir_at = next(i for i, ln in code if "mkdir -p" in ln)
    first_sbatch = next(i for i, ln in code if re.search(r"sbatch\s+--", ln))
    assert mkdir_at < first_sbatch, "log dir must be created before the first sbatch"
