# GNN_CaCoSE

A from-the-paper reimplementation of **CaCoSE** — Cross-attentive Cohesive Subgraph Embedding,
which mitigates oversquashing in GNNs by decomposing a graph into cohesive subgraphs via k-core
edge scoring, embedding each independently, and recovering long-range structure with attention
across the subgraphs.

> Hossain, Islam, Chebbah, Fanning and Akbas. *Cross-attentive Cohesive Subgraph Embedding to
> Mitigate Oversquashing in GNNs.* arXiv:2603.27529v3.

No reference implementation is publicly available, so everything here is derived from the paper
text, its Figure 2 worked example, and Algorithm 1 in its appendix.

## Status

**Phase 1 complete — all three reproduction targets met.**

| Phase | Goal | State |
|---|---|---|
| 1 | Reproduce the paper on Cora, Chameleon, MUTAG | **Complete** — all three pass; see `REPRO_REPORT.md` |
| 2 | Apply the method to a new domain | Not started |
| 3 | Attention visualiser | Not started |

Reproduction results (paper Tables 1–2, mean ± std over 10 seeds):

| Dataset | Task | Paper | Accept at | Ours (10 seeds) |
|---|---|---|---|---|
| Cora | Node classification | 85.00 | ≥ 83.5 | **84.62 ± 1.91** |
| Chameleon | Node classification | 68.99 | ≥ 66.5 | **67.25 ± 2.24** |
| MUTAG | Graph classification | 76.99 | ≥ 74.0 | **82.50 ± 8.58** |

## Layout

```
REPRO_REPORT.md        the results, the findings, and the questions for the author
RUNBOOK.md             running it on the AART pleiades cluster
plans/phase1.md        engineering plan - milestones, APIs, config schema
writeups/
  phase1_implementation.tex   design spec - method, equations, ambiguity log
CaCoSE.pdf             the source paper

cacose/
  types.py             Subgraph, Decomposition, Splits, RunResult, GraphSample
  registry.py          name -> class registry, used by all five swappable families
  config.py            typed config; YAML with `extends`; seed-independent config hash
  decompose/           k-core edge scoring, CaEF, edge-set partition, disk cache
  nn/                  backbone, pooling, readout, cross-subgraph attention, merge, model
  tasks/               node and graph classification behind one interface
  data/                dataset providers and split strategies
  eval/                metrics and evaluator
  engine/              paths, trainer, experiment runner
  results.py           result store and aggregation

configs/               base.yaml + one per dataset (+ mutag_cv.yaml for k-fold)
scripts/               run.py, sweep_seeds.py, prefetch_data.py, submit.sh
slurm/                 cacose.def, build.sh, prefetch.sh, run_seeds.sbatch
tests/                 322 tests, CPU-only and offline
results/               one JSON per (dataset, config, seed)   (gitignored)
```

Five families are registry-backed — decomposer, dataset provider, split strategy, pooling,
backbone — so the paper's Table 4 ablations and Figure 5 decomposition comparison are config
changes rather than code changes.

Each phase gets a pair: a spec in `writeups/` (what and why — stable, edited only when a design
decision changes) and a plan in `plans/` (how and in what order — edited every milestone).

## What we found

Three things the paper leaves open turned out to matter. Full detail in `REPRO_REPORT.md`; these
are the ones worth putting to the first author.

- **The readout is unspecified and worth 13 accuracy points.** The paper says only "READOUT". On
  MUTAG, `sum` gives 82.50 against 69.00 for `mean+max`, and macro-F1 79.09 against 60.66 — the
  difference between learning both classes and leaning on the majority.
- **The reported MUTAG number is unreachable under the paper's own stated protocol.** With 188
  graphs an 80/10/10 split leaves 18–20 test graphs, so any 10-seed mean is a multiple of
  `1/(10n)`; 76.99 is not one for any n in that range. 10-fold CV, which the paper's own baselines
  use, does admit it.
- **On MUTAG the method's contribution never engages.** Every molecule has `kmax = 2` while δ = 3,
  so CaEF never fires and the decomposition is plain k-core. Chameleon (`kmax = 63`) is where the
  method is actually exercised — and it reproduces.

## Reading the design spec

The PDF is a build artifact and is not tracked. Build it with a TeX distribution:

```bash
cd writeups && latexmk -pdf phase1_implementation.tex
```

Start with §3 (the ambiguity log — every point where the paper underdetermines the implementation,
and what we chose) and §4 (the Figure 2 worked example, which doubles as the primary test fixture).

## Quickstart

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"

pytest                                                        # unit tests, CPU, offline
python -m scripts.run --config configs/cora.yaml --seed 0     # one run
python -m scripts.sweep_seeds                                 # mean +/- std vs the paper
```

Output location is controlled by two environment variables, so the same commands work locally
and on the cluster:

| variable | meaning | default |
|---|---|---|
| `CACOSE_DATA_ROOT` | where datasets are downloaded to | `./data` |
| `CACOSE_OUT` | where `results/`, `logs/` and `cache/` are written | `./` |

Results land at `results/<dataset>/<config_hash>/seed<NN>.json`. The config hash covers everything
except the seed, so a sweep that varies one flag never overwrites the run it is compared against.

GPU training runs on the AART lab `pleiades` cluster via SLURM; see `RUNBOOK.md` once Milestone 5
lands.
