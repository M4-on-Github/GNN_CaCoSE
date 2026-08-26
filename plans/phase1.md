# CaCoSE Phase 1 — engineering plan

**How and in what order.** The *what and why* — method, equations, ambiguity resolutions, the
worked example, the experimental protocol — lives in the design spec:

```
writeups/phase1_implementation.tex     build: cd writeups && latexmk -pdf phase1_implementation.tex
```

The PDF is not tracked (see `writeups/.gitignore`), so build it locally to read it. When the two
documents disagree, the spec wins on design and this file wins on implementation. An equation or a
justification belongs there; a signature, config key, or command belongs here.

---

## Milestones

| | Milestone | Acceptance gate |
|---|---|---|
| [x] | **0a** Design spec | Compiles clean; assumption table signed off |
| [ ] | **0b** Scaffold + local venv | `pip install -e .`, `ruff check .`, `pytest` all run |
| [ ] | **1** Decomposition | `pytest tests/test_decompose.py -q` green |
| [ ] | **2** Layers | `pytest tests/test_layers.py -q` green |
| [ ] | **3** Merge + model | `pytest tests/test_merge.py tests/test_model_smoke.py -q` green |
| [ ] | **4** Data + training | `python -m scripts.run --config configs/cora.yaml --seed 0 --epochs 2` completes |
| [ ] | **5** Cluster handoff | You can run `RUNBOOK.md` top to bottom unaided |
| [ ] | **6** Reproduction | Three 10-seed means clear the spec's §1 thresholds |

Milestones are strictly ordered. Do not start N+1 until N's gate passes.

## Execution split

I build and unit-test locally on CPU. **I do not touch `pleiades`.** All GPU training is submitted
by you; I produce the container definition, the sbatch scripts, and `RUNBOOK.md`, then write
`REPRO_REPORT.md` from the `results/*.json` files you send back.

## Open questions

Spec §3 logs ten ambiguities; eight are Open. Only three affect what gets built, and all three are
handled by config flags with a paper-faithful default, so **none of them block work**:

- #1 `caef_mode`, #3 `share_weights`, #4 `readout`, #5 `attn_residual` — flags, defaults set to the
  most literal reading of the paper.
- #6a, #6b, #7, #8 — settled by decisions already recorded in the spec table; nothing to decide.
- #9 (GCN depth) and #10 ($d_S$) — fixed at 2 and 128; revisit only if Milestone 6 misses.

---

## Configuration schema

One `configs/base.yaml` holds defaults; per-dataset files override. Values marked **frozen** are
fixed by the paper and are never swept.

```yaml
# --- decomposition -------------------------------------------------
delta:          3          # frozen. CaEF threshold
caef_mode:      single     # single | cascade                (ambiguity #1)

# --- model ---------------------------------------------------------
hidden:         128        # frozen. GCN width
d_s:            128        # frozen. subgraph embedding width (#10)
gcn_layers:     2          # (#9)
share_weights:  false      # one GCN_k per subgraph, or one shared  (#3)
pool_ratio:     0.5        # frozen. SAGPool keep-ratio
readout:        mean+max   # mean | max | mean+max | sum     (#4)
num_heads:      2          # frozen. 2 for NC, 1 for GC
attn_residual:  false      # residual + LayerNorm on attention (#5)
dropout:        0.5

# --- training ------------------------------------------------------
lr:             2.5e-3     # frozen
weight_decay:   1.0e-4     # frozen
epochs:         250        # frozen. 250 NC / 100 GC
patience:       50         # frozen. 50 NC / 25 GC

# --- data ----------------------------------------------------------
dataset:        cora
task:           nc         # nc | gc
split_source:   random_per_seed
seeds:          [0,1,2,3,4,5,6,7,8,9]
```

Milestone 6 may vary only `caef_mode`, `readout`, `attn_residual`, `share_weights`, `dropout` —
in that order. Everything marked frozen stays put even if a target is missed.

## Results schema

Each run writes `results/{dataset}_{seed}.json`. `scripts/sweep_seeds.py` and `REPRO_REPORT.md`
both consume this, so the field names are load-bearing:

```json
{
  "dataset": "cora", "task": "nc", "seed": 0,
  "config_hash": "sha1 of the resolved config",
  "best_val_acc": 0.0, "test_acc": 0.0,
  "epochs_run": 0, "best_epoch": 0, "wall_time_s": 0.0,
  "kmax": 0, "num_subgraphs": 0, "num_isolated_nodes": 0,
  "num_params": 0, "git_sha": "", "torch_version": "", "pyg_version": ""
}
```

---

## Milestone 0b — scaffold and local venv

Layout: `cacose/`, `configs/`, `scripts/`, `slurm/`, `tests/`, `results/`, `writeups/`.

**The local venv mirrors the cluster container**, so "green locally" means something for your runs:
`uv venv --python 3.11`, then `torch==2.4.0` (CPU wheel), `torch_geometric==2.6.1`, `networkx`,
`pyyaml`, `pytest`, `ruff`. Your conda base is Python 3.13 with a torch 2.12 nightly on cu128
(required for the 5070's sm_120) — a different world from the cluster's cu121, so the project gets
its own isolated venv rather than borrowing base.

**No `torch_scatter` / `torch_sparse` / `pyg_lib`, deliberately.** PyG ≥ 2.4 has pure-torch
fallbacks for everything used here (`GCNConv`, `SAGPooling`, `global_*_pool`, `scatter`). Those
three compiled packages are the most common cause of a failed Apptainer build, and since you run
the build and I cannot debug it live, dropping them removes the main failure mode at no functional
cost.

`pyproject.toml` carries `[project]` metadata, the `cacose` package, `ruff` + `pytest` config, and a
`[project.optional-dependencies] dev` extra holding `pytest` and `ruff` — so `pip install -e ".[dev]"`
is the single install command the README and CI both use.

## Milestone 1 — `cacose/decompose.py`

Pure CPU, NetworkX-backed, deterministic, cached. Method and equations: spec §2.1.

```python
def decompose(edge_index: Tensor, num_nodes: int, *,
              delta: int = 3, caef_mode: str = "single") -> Decomposition
def load_or_compute(dataset: str, edge_index, num_nodes, *, delta, caef_mode,
                    cache_dir: Path) -> Decomposition
```

Implementation notes not in the spec:

- Use `networkx.core_number`. It **raises on self-loops** — strip them before the call.
- Group edges by score, build `G_k` once per distinct `k ≥ delta`, and check support there. Do not
  rebuild `G_k` per edge.
- `caef_mode="cascade"` re-checks at `k−1` while `k−1 ≥ delta`; `"single"` does not.
- Cache key `(dataset, delta, caef_mode)`; store as `.pt`. Document the format in `README.md` — it
  is the Phase 2 entry point.

**Tests (`tests/test_decompose.py`).** The Figure 2 fixture is specified in spec §4; assert core
numbers, per-edge raw and final scores, the `{S_2, S_3, S_4}` partition sizes, and `v4 ∈ {S_2, S_4}`.

Property tests over 30 `networkx.gnp_random_graph` samples:

```python
assert every input edge appears in exactly one S_k
assert set(union of node_map_k) == set of non-isolated nodes
assert (support_at_raw_k == 0) == (final_score == raw_k - 1)   # for raw_k >= delta
assert decompose(delta=kmax+1) == raw k-core partition          # CaEF is a no-op
assert cascade == single  when no edge is demoted twice
```

The third assertion is the correct invariant; the tempting simpler one is false. Spec §6 explains
why.

## Milestone 2 — `cacose/layers.py`

- **`GCNBlock(in_dim, hidden, num_layers=2, dropout)`** — `GCNConv`, ReLU between layers. One block
  per `k` in an `nn.ModuleDict` keyed `str(k)`; `share_weights=True` collapses to a single block.
- **`SAGPoolBlock(hidden, ratio, readout, d_s)`** — `SAGPooling(hidden, ratio=ratio, GNN=GCNConv)`.
  GCN, **not** PyG's default `GraphConv`, because the paper scores with a GCN. Readout per config,
  then `Linear → d_s` when the readout width differs.
- **`CrossSubgraphAttention(d_s, num_heads, residual=False)`** —
  `nn.MultiheadAttention(d_s, num_heads, batch_first=True)` with Q=K=V=`Z_S`. Call with
  **`average_attn_weights=False`** so per-head weights survive for the Phase 3 visualiser. Returns
  `(Z_S_attn, attn_weights)`.

**Tests:** output shapes; attention rows sum to 1; with `num_heads=1` and identity projections,
output equals the softmax-weighted average of `Z_S` (numerical check on a tiny input).

## Milestone 3 — `cacose/merge.py`, `cacose/model.py`

**Merge.** Per subgraph, `h_v_k = cat([H_k[local], Z_k_attn.expand(...)], dim=-1)`, then
`Z_v.index_add_(0, node_map_k, h_v_k)` into a `zeros(num_nodes, hidden + d_s)`. Isolated nodes keep
zeros; count them into `num_isolated_nodes`.

```python
def merge_nodes(H: list[Tensor], Z_attn: Tensor,
                decomposition: Decomposition) -> tuple[Tensor, int]
```

**Test:** with synthetic `H_k` on the Figure 2 fixture, `z_v4 == h_v4(2) + h_v4(4)` exactly, and a
node in one subgraph equals that single row.

**Model.**

```python
class CaCoSE(nn.Module):
    def __init__(self, in_dim, hidden, d_s, num_classes, decomposition, task, *,
                 num_heads, pool_ratio, share_weights, readout, attn_residual,
                 gcn_layers, dropout)
    def forward(self, x, decomposition_or_batch) -> tuple[Tensor, dict]
```

The `aux` dict is a fixed contract (spec §2.6) and gets its own test asserting keys and shapes.

**Graph-classification batching.** Decompose every graph once in `data.py`; store subgraphs as a
flat list of `Data` objects carrying `graph_id` and `k`; PyG's `DataLoader` collates them; attention
runs per `graph_id` group with padding and a `key_padding_mask`, since each graph has its own `N_S`.
The `ModuleDict` spans the union of `k` across the dataset.

**Smoke test:** forward + backward for both tasks, finite loss, no `None` grads on any `GCN_k`
exercised by the batch. Scope that assertion to the `k` values actually present — a block for an
absent `k` legitimately receives no gradient.

## Milestone 4 — `cacose/data.py`, `cacose/train.py`

```python
def load_dataset(name: str, seed: int, cfg: dict) -> tuple[Data | list[Data], Splits]
def train(cfg: dict, seed: int) -> dict          # returns the results-schema payload
```

Preprocess with `to_undirected` then `remove_self_loops` before decomposing; `GCNConv` re-adds
self-loops internally, so this does not affect the model.

| Dataset | Constructor | Split |
|---|---|---|
| Cora | `Planetoid('Cora')` | per-seed permutation → 500 val / 500 test / 1208 train (500 unused) |
| Chameleon | `WikipediaNetwork('chameleon', geom_gcn_preprocess=True)` | built-in masks **ignored**; per-seed stratified 48/32/20 |
| MUTAG | `TUDataset('MUTAG')` | per-seed stratified 80/10/10 |

Dataset root from `$CACOSE_DATA_ROOT`, default `./data`.

Training: cross-entropy, `Adam`, early stop on validation accuracy, restore best-val weights, then
evaluate test once. Seed `torch`, `numpy`, `random`, and `torch_geometric.seed_everything`; set
`torch.use_deterministic_algorithms(True, warn_only=True)`.

`scripts/sweep_seeds.py` aggregates `results/*.json` into a markdown mean ± std table, diffed
against the spec's §1 targets.

## Milestone 5 — cluster handoff

`slurm/cacose.def` — from `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`; installs
`torch_geometric==2.6.1 networkx pyyaml pytest ruff` and `pip install -e .`. No `torch_scatter` /
`torch_sparse` / `pyg_lib` (Milestone 0b). cu121 covers pleiades-0-17's RTX 6000 Ada (sm_89).

`slurm/run_seeds.sbatch` — array 0–9, `--partition=pleiades`, `--nodelist=pleiades-0-17`, 4 CPUs,
1 GPU, 16 GB, 2 h wall.

`scripts/prefetch_data.py` + `RUNBOOK.md`. Compute nodes commonly have no outbound network, so PyG's
first-use download would fail inside the array job. The runbook has you prefetch datasets into
`/data/$USER/datasets` once, build the `.sif` in an interactive session, then submit — copy-pasteable
commands with expected output at each step.

## Milestone 6 — reproduction

Run order Cora → Chameleon → MUTAG, 10 seeds each. Targets: spec §1, Table 1.

You send back `results/*.json`; I aggregate and write `REPRO_REPORT.md`. If a target is missed I
propose the next config from the frozen sweep order above and you run it; every configuration tried
is logged in `results/REPRO_LOG.md`.

**Chameleon is the expected trouble spot** (spec §6 has the analysis). If it misses, try
`share_weights=true` first.

`REPRO_REPORT.md` carries the mean ± std table, the winning config, per-dataset `kmax` and subgraph
counts, wall time, and the ambiguity log for the first author.

---

## Continuous verification

- `pytest` green in the local CPU venv — decomposition, layers, merge, model smoke, `aux` contract.
- `ruff check .` clean.
- GitHub Actions on `M4-on-Github/GNN_CaCoSE`: CPU-only, installs the venv deps, runs `ruff` and
  `pytest` on push and PR.
- A seconds-long CPU smoke run before every handoff, to prove the entrypoint works end to end.
- `pip install -e .` inside the Apptainer image — verified by you during the runbook.
