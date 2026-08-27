# CaCoSE — Phase 1 reproduction report

**Status: in progress.** Cora complete; Chameleon and MUTAG running. Every `TBD` below is filled
from `results/` by `python -m scripts.sweep_seeds`.

A from-the-paper reimplementation of Hossain, Islam, Chebbah, Fanning and Akbas,
*Cross-attentive Cohesive Subgraph Embedding to Mitigate Oversquashing in GNNs*
(arXiv:2603.27529v3). No reference implementation is publicly available, so every construction
is derived from the paper text, its Figure 2, and Algorithm 1 in its appendix. The design is
fixed in `writeups/phase1_implementation.tex`; this document reports what happened when it ran.

---

## 1. Headline

| Dataset | Task | Paper | Accept at | Ours (10 seeds) | Δ |
|---|---|---:|---:|---|---:|
| Cora | Node classification | 85.00 | ≥ 83.5 | **84.62 ± 1.91** | −0.38 |
| Chameleon | Node classification | 68.99 | ≥ 66.5 | TBD | TBD |
| MUTAG | Graph classification | 76.99 | ≥ 74.0 | TBD | TBD |

Mean ± sample standard deviation over seeds 0–9. Accept thresholds are from the design spec §1
and were fixed before any run.

## 2. Configuration

Hyperparameters are the paper's §4.1 verbatim; nothing was tuned. The only settings not taken
from the paper are the ones it does not state — see the ambiguity log in §6.

| | Cora | Chameleon | MUTAG |
|---|---|---|---|
| config hash | `4c87d394` | TBD | TBD |
| δ / caef_mode | 3 / single | 3 / single | 3 / single |
| backbone / pooling / readout | gcn / sagpool / mean+max | same | same |
| share_weights / attn_residual | false / false | false / false | false / false |
| heads | 2 | 2 | 1 |
| dropout | 0.5 | 0.5 | 0.0 (see #11) |
| epochs / patience | 250 / 50 | 250 / 50 | 100 / 25 |
| split | 1208/500/500 | 48/32/20 stratified | 80/10/10 stratified |

lr 2.5e-3, weight decay 1e-4, hidden 128, `d_S` 128, pooling ratio 0.5 throughout.

## 3. Decomposition

What the method actually produced on each dataset — the structural facts that drive cost.

| Dataset | `kmax` | subgraphs | isolated nodes | parameters |
|---|---:|---:|---:|---:|
| Cora | 4 | 4 | 0 | 1,032,207 |
| Chameleon | 63 | 50 | TBD | 17,462,889 |
| MUTAG | 2 | 376 (2 per graph) | TBD | 183,942 |

Chameleon is the outlier and was flagged as a risk in the spec before any run: 63 core levels
means ~50 separate GCN and pooling branches, and with a 2325-dimensional input that is 17.5M
parameters fitted to 2277 nodes.

## 4. Cost

| Dataset | s/seed | total | device |
|---|---:|---:|---|
| Cora | 7.4 | 1.2 min | RTX (cluster) |
| Chameleon | TBD | TBD | |
| MUTAG | TBD | TBD | |

## 5. Observations

**Cora reproduces.** −0.38 points against a seed spread of ±1.91 is comfortably inside noise.

**Seed variance is large and the paper reports none.** Cora's ten seeds span 81.2 to 87.8 — a
6.6-point range on a dataset usually treated as stable. The paper gives a bare 85.00 with no
dispersion, so there is no way to tell whether it is a comparable mean or a favourable draw.
Any single-seed comparison against this method is close to uninformative. **Worth raising with
the first author.**

**CaEF mode is not a detail.** Single versus cascade produces a *different partition on 24 of 30*
random graphs at δ = 3. Ambiguity #1 is therefore the highest-value question in §6, not a
footnote.

**Dropout is not in the paper at all, and it matters.** At 0.5, MUTAG's evaluation loss diverges
from its training loss (0.47 → 1.03) and predictions collapse to a single class; at 0.0 they do
not. The mechanism: dropout perturbs the features SAGPool selects nodes from, so the model never
learns the deterministic eval-time selection. Brutal on MUTAG's ~18-node graphs, harmless on
Cora. Recorded as ambiguity #11.

TBD — Chameleon and MUTAG observations once their sweeps land.

## 6. Ambiguity log — for the first author

Eleven points where the paper underdetermines the implementation. The canonical version lives in
the design spec §3; this is the same table in a form that can be sent directly. Two are settled
by the paper's own evidence; the rest are working assumptions.

| # | Question | What we implemented | Status |
|---|---|---|---|
| 1 | CaEF decrement: once, or cascade? | Single decrement, configurable | **Open — highest value** |
| 2 | Triadic support inside `G_k` or in `G`? | Inside `G_k` | **Resolved** by Figure 2 |
| 3 | `GCN_k` weights separate or shared? | Separate | Open |
| 4 | SAGPool readout function | mean ‖ max | Open |
| 5 | Residual/LayerNorm on attention? | Absent | Open |
| 6a | Cora split fixed or per seed? | Re-randomised per seed | Open |
| 6b | Cora 1208/500/500 leaves 500 nodes unused | Implemented literally | Open |
| 7 | Chameleon version and split | Geom-GCN graph, our own 48/32/20 | **Resolved** by the paper's text |
| 8 | Meaning of "1-hot encoding" features | Native dataset features | Open |
| 9 | GCN depth per subgraph | Two layers | Open |
| 10 | Is `d_S` the GCN hidden width? | Both 128 | Open |
| 11 | Dropout rate — **never stated** | 0.5 NC, 0.0 GC | Open |

**On #2.** The paper's Figure 2 states that edge (v4,v7) "has no support" at k = 3. In the full
graph its endpoints share neighbour v6, so support computed on `G` would be 1 and the edge would
survive. But core(v6) = 2, so v6 is absent from `G_3`. The figure is only consistent if support
is counted inside `G_k`. This is encoded as the primary unit-test fixture.

## 7. Reproducing this

```bash
git clone https://github.com/M4-on-Github/GNN_CaCoSE.git && cd GNN_CaCoSE
uv venv --python 3.11 && uv pip install -e ".[dev]"
pytest

python -m scripts.run --config configs/cora.yaml --seed 0
python -m scripts.sweep_seeds
```

On the AART `pleiades` cluster, follow `RUNBOOK.md`. Every result records the commit it came
from, the resolved config, and the torch/PyG versions, so any row above can be traced back.

Commit: TBD (filled at completion) · torch 2.5.1+cu121 · PyG 2.6.1 · NumPy 1.26
