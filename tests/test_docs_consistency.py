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


SCRIPTS = ["train_benchmark_array.sbatch", "build_container.sbatch"]


def script_text(name: str) -> str:
    return (REPO / "slurm" / name).read_text(encoding="utf-8")


def directives(name: str) -> str:
    lines = [ln for ln in script_text(name).splitlines() if ln.startswith("#SBATCH")]
    return "\n".join(lines)


def test_runbook_and_container_agree_on_paths():
    runbook = (REPO / "RUNBOOK.md").read_text(encoding="utf-8")
    sbatch = script_text("train_benchmark_array.sbatch")
    for fragment in ("/data/", "containers/cacose.sif", "datasets"):
        assert fragment in runbook and fragment in sbatch, f"{fragment} missing from one of them"


def test_sweep_keeps_the_concurrency_cap_and_no_node_pin():
    """Two throttles were a deliberate decision; losing either changes cluster impact."""
    d = directives("train_benchmark_array.sbatch")
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

    submit_benchmark_sweep.sh hashes cacose.def, compares it against the hash recorded beside the image, and
    chains the sweep behind a rebuild with --dependency=afterok when they differ.
    """
    submit = (REPO / "scripts" / "submit_benchmark_sweep.sh").read_text(encoding="utf-8")
    build = (REPO / "slurm" / "build_container.sbatch").read_text(encoding="utf-8")

    assert "sha256sum" in submit and "def.sha256" in submit
    assert "--dependency=afterok" in submit
    assert "--hold" not in submit, "sweeps run unattended; %3 is the only throttle"
    # the build must write the sidecar, or submit_benchmark_sweep.sh would rebuild on every invocation
    assert "def.sha256" in build and "sha256sum" in build


def test_submit_creates_the_log_dir_before_sbatch_opens_it():
    """sbatch fails outright if --output names a directory that does not exist.

    Compares line numbers of real commands: the word "sbatch" also appears in the comments
    explaining this very requirement, so a plain substring search finds prose, not a call.
    """
    lines = (REPO / "scripts" / "submit_benchmark_sweep.sh").read_text(encoding="utf-8").splitlines()
    code = [(i, ln) for i, ln in enumerate(lines) if not ln.strip().startswith("#")]
    mkdir_at = next(i for i, ln in code if "mkdir -p" in ln)
    first_sbatch = next(i for i, ln in code if re.search(r"sbatch\s+--", ln))
    assert mkdir_at < first_sbatch, "log dir must be created before the first sbatch"


@pytest.mark.parametrize(
    "script",
    [
        "scripts/submit_benchmark_sweep.sh",
        "scripts/submit_all_benchmarks.sh",
        "slurm/build_container.sbatch",
        "slurm/download_datasets.sbatch",
        "slurm/train_benchmark_array.sbatch",
    ],
)
def test_shell_scripts_are_executable_in_git(script):
    """`scripts/submit_benchmark_sweep.sh ...` fails with Permission denied unless git stores mode 100755.

    chmod on Windows does not reach the index -- git there ignores the filesystem exec bit --
    so this has to be set with `git update-index --chmod=+x` and guarded here.
    """
    import subprocess

    out = subprocess.check_output(["git", "ls-files", "-s", script], text=True, cwd=REPO)
    mode = out.split()[0]
    assert mode == "100755", f"{script} is {mode}; run: git update-index --chmod=+x {script}"


def _ambiguity_section(report_path) -> str:
    """Just the ambiguity log, not the whole report.

    Other tables in the report have numeric first columns too -- the test-graph arithmetic in
    the MUTAG section is one -- so matching table rows document-wide picks up rows like 18, 19
    and 188 and reports a mismatch that is not real.
    """
    text = report_path.read_text(encoding="utf-8")
    start = text.index("## 6. Ambiguity log")
    end = text.index("## 7.", start)
    return text[start:end]


def test_report_ambiguity_table_matches_the_spec():
    """REPRO_REPORT.md restates the spec's ambiguity log so it can be sent to the author on its
    own. Two copies of one table is exactly how drift starts, so the row ids must agree."""
    report = REPO / "REPRO_REPORT.md"
    if not report.exists():  # pragma: no cover
        pytest.skip("report not written yet")
    rows = set(re.findall(r"^\| (\d+[ab]?) \|", _ambiguity_section(report), re.M))
    assert rows == ambiguity_ids(), (
        f"report and spec disagree; only in report: {sorted(rows - ambiguity_ids())}, "
        f"only in spec: {sorted(ambiguity_ids() - rows)}"
    )


def test_report_marks_the_same_items_resolved_as_the_spec():
    report = REPO / "REPRO_REPORT.md"
    if not report.exists():  # pragma: no cover
        pytest.skip("report not written yet")
    text = _ambiguity_section(report)
    for row_id in ("2", "7"):
        line = next(ln for ln in text.splitlines() if ln.startswith(f"| {row_id} |"))
        assert "Resolved" in line, f"ambiguity #{row_id} is Resolved in the spec but not here"


def test_every_script_named_in_the_runbook_exists():
    """A runbook that names a file which was since renamed is worse than no runbook."""
    runbook = (REPO / "RUNBOOK.md").read_text(encoding="utf-8")
    named = set(re.findall(r"(?:scripts|slurm)/[a-z_]+\.(?:sh|py|sbatch|def)", runbook))
    assert named, "the runbook should name the scripts it tells you to run"
    missing = [n for n in sorted(named) if not (REPO / n).exists()]
    assert not missing, f"RUNBOOK.md names files that do not exist: {missing}"


def test_sbatch_job_names_match_their_filenames():
    """`squeue` shows the job name, so it should point at the file that produced it."""
    expected = {
        "build_container.sbatch": "cacose-build",
        "download_datasets.sbatch": "cacose-datasets",
        "train_benchmark_array.sbatch": "cacose",
    }
    for filename, job_name in expected.items():
        text = (REPO / "slurm" / filename).read_text(encoding="utf-8")
        found = re.search(r"^#SBATCH (?:-J|--job-name=)\s*(\S+)", text, re.M)
        assert found, f"{filename} sets no job name"
        assert found.group(1) == job_name, f"{filename}: job name is {found.group(1)!r}"


def test_submit_sweep_writes_only_the_job_id_to_stdout():
    """--parsable exists so submit_all_benchmarks.sh can chain --dependency on the job id.

    Any status line reaching stdout is captured as part of that id, and SLURM then rejects the
    next submission with the opaque "Job dependency problem" -- which is exactly what happened
    the first time this ran on the cluster. Status must go to stderr.
    """
    text = (REPO / "scripts" / "submit_benchmark_sweep.sh").read_text(encoding="utf-8")
    body = text[text.index("# ── container freshness") :]

    offenders = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("echo ", "printf ", "cat <<")):
            continue
        if ">&2" in stripped or stripped.startswith("say "):
            continue
        if stripped == 'echo "$JOBID"':  # the one legitimate stdout write
            continue
        # the human-readable summary only runs when not parsable, which exits before it
        if "PARSABLE" in body[: body.index(line)].rsplit("if", 1)[-1]:
            continue
        offenders.append(stripped)

    assert 'echo "$JOBID"' in body, "parsable mode must emit the job id"
    assert "say()" in text, "status helper must exist so messages default to stderr"
    early = body[: body.index('echo "$JOBID"')]
    leaked = [
        ln.strip()
        for ln in early.splitlines()
        if ln.strip().startswith("echo ") and ">&2" not in ln
    ]
    assert not leaked, f"status written to stdout before the job id: {leaked}"


def test_submit_all_validates_the_job_id_before_chaining():
    """A malformed id must fail with an explanation, not with SLURM's dependency error."""
    text = (REPO / "scripts" / "submit_all_benchmarks.sh").read_text(encoding="utf-8")
    assert "^[0-9]+$" in text, "the chained job id must be validated as numeric"


def test_build_verification_retries_and_never_fails_the_job():
    """A transient exec failure on /data must not cancel every benchmark behind the build.

    The image is on network storage, and exec'ing it immediately after writing can return
    "input/output error" while the file settles. That once failed the build job, leaving three
    chained arrays in DependencyNeverSatisfied even though the container was fine.
    """
    text = (REPO / "slurm" / "build_container.sbatch").read_text(encoding="utf-8")
    assert "for attempt in" in text, "verification must retry"
    assert "VERIFIED" in text and "WARNING" in text, "a failed verify must warn, not exit"
    verify_block = text[text.index("Verifying the stack") :]
    assert "exit 1" not in verify_block, "verification must not fail the build job"


def test_sweep_is_cancelled_if_its_build_fails():
    """Without --kill-on-invalid-dep a failed build leaves the sweep parked forever."""
    text = (REPO / "scripts" / "submit_benchmark_sweep.sh").read_text(encoding="utf-8")
    assert "--kill-on-invalid-dep=yes" in text


@pytest.mark.parametrize(
    "script",
    [
        "slurm/build_container.sbatch",
        "slurm/download_datasets.sbatch",
        "slurm/train_benchmark_array.sbatch",
        "scripts/submit_benchmark_sweep.sh",
        "scripts/submit_all_benchmarks.sh",
    ],
)


def _command_lines(script: str) -> str:
    """Script text with comments stripped -- prose mentions paths that code never touches."""
    text = (REPO / script).read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    return "\n".join(lines)


CLUSTER_SCRIPTS = [
    "slurm/build_container.sbatch",
    "slurm/download_datasets.sbatch",
    "slurm/train_benchmark_array.sbatch",
    "scripts/submit_benchmark_sweep.sh",
    "scripts/submit_all_benchmarks.sh",
]


@pytest.mark.parametrize("script", CLUSTER_SCRIPTS)
def test_nothing_writes_outside_the_users_own_data_directory(script):
    """`/data/shared` is the lab's shared area, read-only for student accounts. Nothing here may
    reference it, and every /data path must sit under the per-user tree."""
    code = _command_lines(script)
    assert "/data/shared" not in code

    # (?<![\w.]) so ./data/... in a relative path is not mistaken for an absolute /data/...
    for match in re.finditer(r"(?<![\w.])/data/[^\s\"';:)}]*", code):
        path = match.group(0)
        assert path.startswith(("/data/$USER/", "/data/%u/", "/data/${USER}/")), (
            f"{script} references {path}, which is outside /data/$USER"
        )


@pytest.mark.parametrize("script", CLUSTER_SCRIPTS[:3])
def test_container_sees_only_the_three_intended_binds(script):
    """--containall means the container sees nothing that is not bound explicitly, so the bind
    list is the real boundary: the repo, the user's own /data tree, and node-local /tmp.
    /data/shared is not among them and is therefore not even visible inside the container."""
    code = _command_lines(script)
    if "apptainer exec" not in code:
        pytest.skip("no container exec in this script")
    assert "--containall" in code, "without --containall the host filesystem leaks in"

    sources = {b.split(":")[0] for b in re.findall(r'--bind\s+"?([^\s"]+)', code)}
    assert sources, "expected at least one --bind"
    allowed = {"/tmp", "$REPO", "${REPO}", "$DATA_DIR", "${DATA_DIR}"}
    assert sources <= allowed, f"{script} binds {sources - allowed}, outside the intended three"
