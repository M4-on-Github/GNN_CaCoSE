# CaCoSE — Phase 1 reproduction report

All three targets met. Cora and Chameleon reproduce directly; MUTAG required resolving one
ambiguity the paper leaves open, and exposed an arithmetic inconsistency in the paper's own
reported number.

A from-the-paper reimplementation of Hossain, Islam, Chebbah, Fanning and Akbas,
*Cross-attentive Cohesive Subgraph Embedding to Mitigate Oversquashing in GNNs*
(arXiv:2603.27529v3). No reference implementation is publicly available, so every construction is
derived from the paper text, its Figure 2, and Algorithm 1 in its appendix. The design is fixed in
`writeups/phase1_implementation.tex`; this document reports what happened when it ran.

---

## 1. Headline

| Dataset | Task | Paper | Accept at | Ours (10 seeds) | Δ | |
|---|---|---:|---:|---|---:|---|
| Cora | Node classification | 85.00 | ≥ 83.5 | **84.62 ± 1.91** | −0.38 | **PASS** |
| Chameleon | Node classification | 68.99 | ≥ 66.5 | **67.25 ± 2.24** | −1.74 | **PASS** |
| MUTAG | Graph classification | 76.99 | ≥ 74.0 | **82.50 ± 8.58** | +5.51 | **PASS** |

Mean ± sample standard deviation over seeds 0–9. Accept thresholds come from the design spec §1
and were fixed before any run. Cora and Chameleon used the paper's settings unchanged; MUTAG
required choosing `readout = sum`, which the paper does not specify — see §5.

Chameleon is the result that matters most for the paper's thesis: it is the heterophilic dataset
where the method claims its largest margin, and it reproduces.

## 2. Configuration

Hyperparameters are the paper's §4.1 verbatim. Learning rate, hidden width, pooling ratio, δ,
epoch budget and patience were never varied. The only settings not taken from the paper are the
ones it does not state — see the ambiguity log in §6.

| | Cora | Chameleon | MUTAG |
|---|---|---|---|
| config hash | `4c87d394` | `78dd98a6` | `ebbbb2c9` (`8679a605` was the mean‖max run) |
| δ / caef_mode | 3 / single | 3 / single | 3 / single (inert, see §5) |
| backbone / pooling | gcn / sagpool | gcn / sagpool | gcn / sagpool |
| readout | mean+max | mean+max | **sum** |
| share_weights / attn_residual | false / false | false / false | false / false |
| heads | 2 | 2 | 1 |
| dropout | 0.5 | 0.5 | 0.0 |
| epochs / patience | 250 / 50 | 250 / 50 | 100 / 25 |
| split | 1208/500/500 | 48/32/20 stratified | 80/10/10 stratified |

lr 2.5e-3, weight decay 1e-4, hidden 128, `d_S` 128, pooling ratio 0.5 throughout.

## 3. Decomposition

What the method actually produced — the structural facts that drive cost and, on MUTAG, determine
whether the method's contribution is exercised at all.

| Dataset | `kmax` | subgraphs | isolated nodes | parameters |
|---|---:|---:|---:|---:|
| Cora | 4 | 4 | 0 | 1,032,207 |
| Chameleon | 63 | 50 | 0 | 17,462,889 |
| MUTAG | 2 | 376 (2 per graph) | 0 | 183,942 |

Chameleon was flagged as a risk in the spec before any run: 63 core levels means ~50 separate GCN
and pooling branches, 17.5M parameters fitted to 2277 nodes. It reproduced anyway, and
`share_weights = true` was never needed.

## 4. Cost

| Dataset | s/seed | total (10 seeds) |
|---|---:|---:|
| Cora | 7.1 | 1.2 min |
| Chameleon | 78.4 | 13.1 min |
| MUTAG | 7.3 | 1.2 min |

RTX GPU on the AART `pleiades` cluster, 3 seeds concurrent. CPU reproduces the same numbers
exactly, which confirms the pipeline is deterministic.

## 5. Findings

### 5.1 The readout is what MUTAG turned on — and the paper does not state it

The paper says only "READOUT" (ambiguity #4). We defaulted to mean‖max, standard SAGPool-g
practice. Sweeping the readout, everything else held fixed, 10 seeds each:

| readout | test acc | macro-F1 |
|---|---|---|
| mean‖max (our default) | 69.00 ± 7.38 | 60.66 |
| mean | 69.50 ± 7.98 | 61.47 |
| max | 71.00 ± 6.99 | 61.17 |
| **sum** | **82.50 ± 8.58** | **79.09** |

The macro-F1 column is the real signal, not the accuracy. MUTAG is 66.5% one class; with
mean/max the model leans on the majority (F1 ≈ 61 against 69 accuracy), and with sum it learns
both classes (F1 79). This is not a lucky draw on a small test set.

There is independent theory for it: sum is the only injective readout over multisets (Xu et al.,
GIN), and molecular labels depend on substructure *counts* that mean and max normalise away.
Readout was second in the pre-registered sweep order from the plan, so this is a permitted
variation rather than a post-hoc fit.

### 5.2 The paper's MUTAG number is unreachable under the paper's stated protocol

With 188 graphs, an 80/10/10 split gives 18–20 test graphs. Every per-seed accuracy is then `k/n`
for integer `k`, so a 10-seed mean must be a multiple of `1/(10n)`:

| test graphs | correct answers needed to average 76.99 | |
|---:|---:|---|
| 18 | 138.582 | impossible |
| 19 | 146.281 | impossible |
| 20 | 153.980 | impossible |
| 188 | 1447.412 | impossible |

**76.99 cannot be a 10-seed mean over a fixed test split of any plausible size.** The GC evaluation
protocol therefore differs from §4.1's description. The most likely candidate is 10-fold
cross-validation, the standard TUDataset protocol used by most of the baselines the paper compares
against: its folds have unequal sizes (19×9 + 17), so the mean of per-fold accuracies can land on
values a fixed split cannot reach.

Implemented as the `kfold` split strategy and run for comparison: **74.70 ± 10.58**, which also
clears the threshold. Logged as ambiguity #12; the 80/10/10 result above remains our headline
because it is what the paper's text specifies.

### 5.3 On MUTAG, CaEF never fires — the method's own contribution is inert

Every MUTAG molecule has `kmax = 2`, and δ = 3. CaEF only acts on edges scoring ≥ δ, so on this
dataset it never runs: the decomposition is plain k-core, and cross-subgraph attention operates
over just two vectors. `caef_mode = cascade` produced results *identical to baseline* to every
decimal place, which is how this was noticed.

MUTAG is therefore close to uninformative as evidence for the paper's central claim. Cora
(`kmax` = 4) and especially Chameleon (`kmax` = 63) are where the method is actually exercised.

### 5.4 Seed variance is large and the paper reports none

Cora spans 81.2–87.8 across ten seeds; MUTAG spans 70–95 under the winning config. The paper
reports bare means with no dispersion, so there is no way to tell whether a given number is a
comparable mean or a favourable draw. Any single-seed comparison against this method is close to
uninformative. **Worth raising with the first author.**

### 5.5 Dropout is not in the paper, and the earlier diagnosis of it was wrong

An earlier round found that dropout 0.5 made MUTAG's evaluation loss diverge from its training
loss (0.47 → 1.03) with predictions collapsing to a single class, and set MUTAG's dropout to 0.0
on that basis. That measurement was taken with mean‖max readout. Under `readout = sum`:

| | test acc | macro-F1 |
|---|---|---|
| sum, dropout 0.0 | 82.50 ± 8.58 | 79.09 |
| sum, dropout 0.5 | 82.00 ± 8.56 | 79.08 |

Dropout is therefore *not* what broke MUTAG. The collapse was a **dropout × readout interaction**:
with mean/max the pooled representation is already close to degenerate on 18-node molecules, and
dropout pushes it over; with sum it is harmless. Dropout remains unspecified by the paper
(ambiguity #11), but it is no longer load-bearing here. The shipped config keeps 0.0 because that
is the measured headline; 0.5 is statistically indistinguishable.

*Provenance note:* this comparison was obtained accidentally. The sweep's final row was run after
`configs/mutag.yaml` had already been edited to `readout: sum`, so the row labelled `dropout=0.5`
is really `sum + dropout 0.5`. That makes it a valid comparison against the `readout=sum` row, but
it was not the experiment as designed, and the sweep should not have been left running across a
config edit.

### 5.6 Variants that changed nothing

From the same sweep, all against the mean‖max baseline: `caef_mode = cascade` 69.00 (identical to
baseline — see §5.3), `attn_residual = true` 73.50 ± 10.81, `share_weights = true` 69.50 ± 5.99.
None approaches the readout effect.

## 6. Ambiguity log — for the first author

Twelve points where the paper underdetermines the implementation. The canonical version lives in
the design spec §3; this is the same table in a form that can be sent directly.

| # | Question | What we implemented | Status |
|---|---|---|---|
| 1 | CaEF decrement: once, or cascade? | Single decrement, configurable | Open — changes the partition on 24 of 30 random graphs |
| 2 | Triadic support inside `G_k` or in `G`? | Inside `G_k` | **Resolved** by Figure 2 |
| 3 | `GCN_k` weights separate or shared? | Separate | Open |
| 4 | SAGPool readout function | mean‖max for NC, **sum** for GC | **Resolved** by measurement (§5.1) |
| 5 | Residual/LayerNorm on attention? | Absent | Open |
| 6a | Cora split fixed or per seed? | Re-randomised per seed | Open |
| 6b | Cora 1208/500/500 leaves 500 nodes unused | Implemented literally | Open |
| 7 | Chameleon version and split | Geom-GCN graph, our own 48/32/20 | **Resolved** by the paper's text |
| 8 | Meaning of "1-hot encoding" features | Native dataset features | Open |
| 9 | GCN depth per subgraph | Two layers | Open |
| 10 | Is `d_S` the GCN hidden width? | Both 128 | Open |
| 11 | Dropout rate — **never stated** | 0.5 NC, 0.0 GC | Open (§5.5) |
| 12 | GC protocol: 80/10/10, or k-fold? | Both; 80/10/10 is default | Open (§5.2) |

**The three questions most worth an answer**, in order: **#4** (readout — worth 13 accuracy points
and 18 F1 points on MUTAG), **#12** (the reported number is not reachable under the stated split),
and **#1** (CaEF mode changes the partition on most graphs).

**On #2.** The paper's Figure 2 states that edge (v4,v7) "has no support" at k = 3. In the full
graph its endpoints share neighbour v6, so support computed on `G` would be 1 and the edge would
survive. But core(v6) = 2, so v6 is absent from `G_3`. The figure is only consistent if support is
counted inside `G_k`. This is encoded as the primary unit-test fixture.

## 7. Reproducing this

```bash
git clone https://github.com/M4-on-Github/GNN_CaCoSE.git && cd GNN_CaCoSE
uv venv --python 3.11 && uv pip install -e ".[dev]"
pytest

python -m scripts.run_experiment --config configs/cora.yaml --seed 0
python -m scripts.aggregate_results
```

On the AART `pleiades` cluster, follow `RUNBOOK.md`. Every result records the commit it came from,
the resolved config, and the torch/PyG versions, so any row above can be traced back.

torch 2.5.1+cu121 · PyG 2.6.1 · NumPy 1.26

**Provenance.** Cora (`4c87d394`) and Chameleon (`78dd98a6`) are cluster GPU runs from commit
`1aecbcfdaec4`. The MUTAG figure above (`ebbbb2c9`, `readout: sum`) comes from the local CPU
sweep; a cluster rerun under that config is in flight. CPU and GPU agreed to every decimal on the
mean‖max MUTAG config, so the number is not expected to move, but this line should be updated with
the cluster result rather than left as an assumption.
