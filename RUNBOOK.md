# RUNBOOK — running CaCoSE on `pleiades`

Copy-pasteable, in order. Steps 0–2 are one-time setup; step 3 onward is per sweep.

**Layout.** Code in `/home/$USER/CaCoSE` (never written to at run time), everything generated in
`/data/$USER/CaCoSE`. The two are bridged by `$CACOSE_DATA_ROOT` and `$CACOSE_OUT`, which
`run_seeds.sbatch` sets for you.

```
/home/mmyatmau/CaCoSE/          git clone — code only
/data/mmyatmau/CaCoSE/
  containers/cacose.sif         built once, step 1
  datasets/                     prefetched, step 2
  cache/decompositions/         written on first run, reused by every later seed
  logs/                         cacose_<jobid>_<task>.out / .err
  results/<dataset>/<confighash>/seed<NN>.json
```

---

## 0. Clone and create the output tree

On the login node:

```bash
ssh mmyatmau@head1.condo.cs.cmu.edu

git clone https://github.com/M4-on-Github/GNN_CaCoSE.git /home/$USER/CaCoSE
mkdir -p /data/$USER/CaCoSE/{containers,datasets,logs,results,cache}
```

Two things about this step:

- **`logs/` must exist before the first submission.** `sbatch` fails immediately if the
  `--output` directory is missing.
- **The clone path is not load-bearing.** The batch scripts derive the repo from
  `$SLURM_SUBMIT_DIR`, so the clone can live anywhere and be named anything -- just submit from
  inside it. `/home/$USER/CaCoSE` is only a convention. Output always goes to
  `/data/$USER/CaCoSE` regardless, via SLURM's `%u` placeholder.

## 1. Build the container — submit it as a job

```bash
cd /home/$USER/CaCoSE
mkdir -p /data/$USER/CaCoSE/logs
sbatch slurm/build_container.sbatch
```

Batch rather than interactive, so the build survives a dropped SSH connection and leaves a log:

```bash
squeue -u $USER
tail -f /data/$USER/CaCoSE/logs/cacose-build_<jobid>.out
```

Takes 5-15 minutes, mostly pulling the base image. A successful log ends with the resolved
versions and `stack OK`.

Notes on what the job does:

- **No GPU requested.** A build unpacks an image and runs pip; it is I/O bound. The definition's
  `%test` therefore prints `cuda available: False`, which is expected here and not a problem --
  CUDA is exercised for real by the first training job.
- **Cache and scratch go to `/data`.** Pulling the CUDA base image writes several GB, and `/home`
  is the quota'd partition.
- **An existing `.sif` is moved aside, not overwritten**, so a failed rebuild cannot leave you
  with nothing.
- If an unprivileged build fails, the script retries with `--fakeroot` before giving up.

### If it reports `no container runtime found`

`apptainer` is not always on the PATH, and on some clusters it is still called `singularity` or
lives behind an environment module. Both `build_container.sbatch` and `run_seeds.sbatch` source
`slurm/_runtime.sh`, which checks the PATH, then `module load`, then the usual install
locations. If it still finds nothing, diagnose with:

```bash
command -v apptainer singularity
module avail 2>&1 | grep -iE 'apptainer|singularity'
ls -1 /usr/local/bin /opt 2>/dev/null | grep -iE 'apptainer|singularity'
```

If the runtime exists only on the login node, build there instead -- a build is I/O bound, so it
is far less objectionable on `head1` than training would be:

```bash
cd /home/$USER/CaCoSE
bash slurm/build_container.sbatch      # the #SBATCH lines are inert when run directly
```

## 2. Prefetch datasets — on the login node, which has network

```bash
cd /home/$USER/CaCoSE
export CACOSE_DATA_ROOT=/data/$USER/CaCoSE/datasets

source slurm/_runtime.sh && require_container_runtime
"$CONTAINER_CMD" exec /data/$USER/CaCoSE/containers/cacose.sif \
    python -m scripts.prefetch_data --all
```

Expected:

```
  OK   cora (planetoid)             graphs=    1 features= 1433 classes=7
  OK   chameleon (wikipedia)        graphs=    1 features= 2325 classes=5
  OK   mutag (tudataset)            graphs=  188 features=    7 classes=2
all datasets present (~50 MB under /data/mmyatmau/CaCoSE/datasets)
```

**This step is not optional.** Compute nodes typically have no outbound network, so a
first-use download inside an array job fails after the job has already queued and been
scheduled. `run_seeds.sbatch` refuses to start if `datasets/` is missing.

## 3. Submit a sweep -- held, and rebuilt first if stale

```bash
cd /home/$USER/CaCoSE
scripts/submit.sh configs/cora.yaml
```

`submit.sh` does three things before anything runs:

1. Creates the log directory, since `sbatch` fails outright if `--output` names a missing path.
2. Hashes `slurm/cacose.def` and compares it against the hash recorded beside the `.sif`. If
   they differ -- or no image exists -- it submits a build job and chains the sweep behind it
   with `--dependency=afterok`, so tasks can only start after a successful build. **This is what
   stops a stale image from quietly producing results under the wrong dependency versions.**
3. Submits the sweep `--hold`, so nothing starts until you release it.

Output when the container is already current:

```
[container] up to date (a1b2c3d4e5f6)
submitted held : job 27310   (configs/cora.yaml)
```

and when it is not:

```
[container] STALE -- cacose.def changed since cacose.sif was built
[container] build job 27309 submitted; the sweep will wait for it
submitted held : job 27310   (configs/cora.yaml)
depends on     : build job 27309 (afterok)
```

```bash
squeue -j <jobid>          # ST=PD, REASON=JobHeldUser (or Dependency)
```

## 4. Release

```bash
scontrol release <jobid>            # all 10 seeds; the array's %3 cap admits 3 at a time
scontrol release <jobid>_0          # or just one seed first, to sanity-check
```

Two independent throttles: `--hold` is your manual gate, `%3` is automatic and still applies
after a blanket release. At most three tasks of this array ever run at once.

Watch:

```bash
squeue -u $USER
tail -f /data/$USER/CaCoSE/logs/cacose_<jobid>_0.out
```

A healthy task ends with a line like:

```
done   : test_acc=0.8560 best_val=0.8520 epochs=40 (best 20) kmax=4 subgraphs=4 params=1,032,207 5.13s
wrote  : /data/mmyatmau/CaCoSE/results/cora/66c03b2a/seed00.json
```

## 5. Collect

```bash
source slurm/_runtime.sh && require_container_runtime
"$CONTAINER_CMD" exec /data/$USER/CaCoSE/containers/cacose.sif \
    env CACOSE_OUT=/data/$USER/CaCoSE python -m scripts.sweep_seeds
```

```
| dataset | config | seeds | mean +/- std | paper | delta |
|---|---|---:|---|---:|---:|
| cora | 66c03b2a | 10 | 8x.xx +/- x.xx | 85.00 | +x.xx |

gate: cora [66c03b2a] 8x.xx vs accept >= 83.5 -> PASS
```

Then send back the JSON files, which are small:

```bash
tar czf cora_results.tgz -C /data/$USER/CaCoSE results/cora
```

Repeat steps 3–5 for `configs/chameleon.yaml` and `configs/mutag.yaml`.

---

## Notes and failure modes

**Which nodes.** No `--nodelist`. Peak usage is roughly 1–2 GB of VRAM even on Chameleon, so any
node in the partition works and the queue is shorter. The container's cu121 covers every GPU the
cluster has: GTX 1080Ti (sm_61), Quadro RTX 6000 (sm_75), RTX 6000 Ada (sm_89).

**Rebuilding after a code change: not needed.** The container holds dependencies only; the repo is
bind-mounted from `/home`. `git pull` is enough. Rebuild only when `pyproject.toml` dependencies
change.

**Decomposition cache.** The first seed of a dataset computes the decomposition and writes it to
`cache/decompositions/`; later seeds load it. It is deterministic and keyed by
`(dataset, decomposer, params)`, so changing `delta` or `caef_mode` produces a different file
rather than a stale hit. Safe to delete at any time.

| Symptom | Cause | Fix |
|---|---|---|
| `sbatch: error: Unable to open file` | `logs/` missing | `mkdir -p /data/$USER/CaCoSE/logs` |
| `apptainer: command not found` | runtime not on PATH | the scripts auto-detect via `slurm/_runtime.sh`; see step 1 |
| `couldn't chdir to ...: going to /tmp` | submitted from outside the clone | `cd` into the repo first; the scripts abort with a clear message |
| Job exits with `container not found` | step 1 not done | build the `.sif` |
| Job exits with `datasets missing` | step 2 not done | run `prefetch_data` on head1 |
| `_ARRAY_API not found` | NumPy 2 reached the image | rebuild; the `%test` block should have caught it |
| `"python": executable file not found in $PATH` | image relies on Docker's ENV PATH, which Apptainer ignores at exec time | handled: the scripts probe for the interpreter via `require_container_python`. A rebuild also fixes it permanently (`%environment` now exports PATH) |
| Tasks pending, `REASON=JobHeldUser` | working as intended | `scontrol release <jobid>` |
| Only 3 tasks running | the `%3` cap | intended; raise it in the `--array` line if you want more |

**Never train on head1.** It is a login node. Step 1 is a batch job and step 5 is a few seconds of
aggregation; step 2 is a download, which is fine there. Only fall back to building on head1 if the
container runtime turns out to exist nowhere else.
