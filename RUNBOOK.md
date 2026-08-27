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
  .cache/                       apptainer image cache; safe to delete
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
sbatch slurm/build.sh
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
- **An existing `.sif` is deleted first.** A rebuild replaces the image rather than layering on
  it; if you need to keep the old one, copy it aside before submitting.
- **Builds always use `--fakeroot`.** Your account is not in `/etc/subuid`, so apptainer falls
  back to a root-mapped namespace -- the `User not listed in /etc/subuid` line in the log is
  expected, not an error.

### If it reports `apptainer: command not found`

Try `singularity` in its place, or `module avail 2>&1 | grep -iE 'apptainer|singularity'` to see
whether it lives behind an environment module.

### A note on the interpreter

Every container call addresses Python absolutely, as `/opt/conda/bin/python3`. A bare `python`
does not work: the base image exposes it through Docker's `ENV PATH`, which Apptainer honours
during `%post` and `%test` but not at `exec` time, so `apptainer exec <sif> python` fails with
`"python": executable file not found in $PATH`.

## 2. Prefetch datasets

```bash
cd /home/$USER/CaCoSE
sbatch slurm/prefetch.sh
tail -f /data/$USER/CaCoSE/logs/cacose-prefetch_<jobid>.out
```

Submitted as a job because **apptainer is not installed on head1** -- it exists only on the
compute nodes. Expect:

```
  OK   cora (planetoid)             graphs=    1 features= 1433 classes=7
  OK   chameleon (wikipedia)        graphs=    1 features= 2325 classes=5
  OK   mutag (tudataset)            graphs=  188 features=    7 classes=2
all datasets present (~50 MB)
```

The datasets must be in place before any sweep: `run_seeds.sbatch` refuses to start without
them, since a mid-job download would fail after the task had already been scheduled.

**If this fails with a network error**, the compute nodes have no outbound access either. Then
fetch the ~50 MB somewhere that does -- your laptop -- and copy it up:

```bash
# locally
python -m scripts.prefetch_data --all --data-root ./data
scp -r ./data/* mmyatmau@head1.condo.cs.cmu.edu:/data/$USER/CaCoSE/datasets/
```

## 3. Submit the sweeps

All three datasets, one at a time:

```bash
cd /home/$USER/CaCoSE
scripts/submit_all.sh
```

Each sweep is 10 seeds capped at 3 concurrent. Submitting all three at once would allow 9, since
each array caps independently, so `submit_all.sh` chains them with `--dependency=afterany` --
never more than 3 of your tasks run at a time. Cora and MUTAG take about a minute each, Chameleon
about 13, so Chameleon goes last.

A single dataset, or a subset:

```bash
scripts/submit.sh configs/cora.yaml
scripts/submit_all.sh configs/mutag.yaml configs/chameleon.yaml
```

Runs unattended. The array's `%3` (`JobArrayTaskLimit`) caps it at three seeds at a time, and
SLURM starts the next as each finishes.

`submit.sh` also hashes `slurm/cacose.def` against the hash recorded beside the `.sif`. If they
differ, it submits a rebuild and chains the sweep behind it with `--dependency=afterok`, so a
stale image cannot quietly produce results under the wrong dependency versions.

Watch it:

```bash
squeue -j <jobid>                                              # 3 running, rest JobArrayTaskLimit
tail -f /data/$USER/CaCoSE/logs/cacose_<jobid>_0.out
```

A healthy task ends with:

```
done   : test_acc=0.8560 best_val=0.8520 epochs=40 (best 20) kmax=4 subgraphs=4 params=1,032,207 5.13s
wrote  : /data/mmyatmau/CaCoSE/results/cora/4c87d394/seed00.json
```

Cancel with `scancel <jobid>`.

## 4. Collect

Aggregation reads JSON only -- no container, no torch -- so it runs on head1:

```bash
cd /home/$USER/CaCoSE
CACOSE_OUT=/data/$USER/CaCoSE python3 -m scripts.sweep_seeds
```

```
| dataset | config | seeds | mean +/- std | paper | delta |
|---|---|---:|---|---:|---:|
| cora | 4c87d394 | 10 | 84.62 +/- 1.91 | 85.00 | -0.38 |

gate: cora [4c87d394] 84.62 vs accept >= 83.5 -> PASS
```

If head1's `python3` is too old for the package, run it inside the container on a compute node
instead:

```bash
srun -p pleiades --time=00:05:00 --pty     apptainer exec --bind /data/$USER/CaCoSE:/data/$USER/CaCoSE     --env CACOSE_OUT=/data/$USER/CaCoSE     /data/$USER/CaCoSE/containers/cacose.sif     /opt/conda/bin/python3 -m scripts.sweep_seeds
```

Then send the JSON back -- it is a few kilobytes:

```bash
tar czf cacose_results.tgz -C /data/$USER/CaCoSE results
```

Repeat steps 3-4 for `configs/mutag.yaml` and `configs/chameleon.yaml`. Chameleon last: it is
the slow one (`kmax=63`, 17.5M parameters).

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
| `apptainer: command not found` | runtime not on PATH | try `singularity`, or `module load apptainer` |
| `couldn't chdir to ...: going to /tmp` | submitted from outside the clone | `cd` into the repo first; the scripts abort with a clear message |
| Job exits with `container not found` | step 1 not done | build the `.sif` |
| Job exits with `datasets missing` | step 2 not done | `sbatch slurm/prefetch.sh` |
| `_ARRAY_API not found` | NumPy 2 reached the image | rebuild; the `%test` block should have caught it |
| `"python": executable file not found in $PATH` | image relies on Docker's ENV PATH, ignored at exec time | use `/opt/conda/bin/python3` explicitly, as every script here does |
| Only 3 tasks running, rest `REASON=JobArrayTaskLimit` | the `%3` cap | intended; raise it in the `--array` line if you want more |

**Never train on head1.** It is a login node. Steps 1-3 are all batch jobs, which is also forced
by apptainer being installed only on the compute nodes. Step 4 is a few seconds of reading JSON
and needs neither the container nor torch, so it runs fine on head1.
