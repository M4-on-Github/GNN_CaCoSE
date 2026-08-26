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

**Phase 1 — design complete, implementation not started.**

| Phase | Goal | State |
|---|---|---|
| 1 | Reproduce the paper on Cora, Chameleon, MUTAG | Spec written; code not started |
| 2 | Apply the method to a new domain | Not started |
| 3 | Attention visualiser | Not started |

Phase 1 reproduction targets (paper Tables 1–2, mean over 10 seeds):

| Dataset | Task | Paper | Accept at |
|---|---|---|---|
| Cora | Node classification | 85.00 | ≥ 83.5 |
| Chameleon | Node classification | 68.99 | ≥ 66.5 |
| MUTAG | Graph classification | 76.99 | ≥ 74.0 |

## Layout

```
phase1.md                          engineering plan — milestones, APIs, config schema
writeups/
  phase1_implementation.tex        design spec — method, equations, ambiguity log
CaCoSE.pdf                         the source paper
cacose/                            the package                        (planned)
  decompose.py                     k-core scoring + CaEF + subgraph extraction
  layers.py  model.py  merge.py    GCN / SAGPool / cross-subgraph attention
  data.py    train.py
configs/  scripts/  tests/         configs, entrypoints, test suite    (planned)
slurm/                             Apptainer definition + sbatch       (planned)
results/                           one JSON per (dataset, seed)        (gitignored)
```

## Reading the design spec

The PDF is a build artifact and is not tracked. Build it with a TeX distribution:

```bash
cd writeups && latexmk -pdf phase1_implementation.tex
```

Start with §3 (the ambiguity log — every point where the paper underdetermines the implementation,
and what we chose) and §4 (the Figure 2 worked example, which doubles as the primary test fixture).

## Quickstart

Not yet available — the package does not exist. Planned:

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
pytest                                                    # unit tests, CPU
python -m scripts.run --config configs/cora.yaml --seed 0
python -m scripts.sweep_seeds                             # mean ± std table
```

GPU training runs on the AART lab `pleiades` cluster via SLURM; see `RUNBOOK.md` once Milestone 5
lands.
