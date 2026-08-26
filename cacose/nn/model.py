"""The CaCoSE model.

Composition, in order: a backbone per subgraph -> pooling per subgraph -> attention *across*
subgraphs -> readouts for nodes and for the graph -> an MLP head.

Two structural facts drive the implementation:

* **Branch count comes from the decomposition, not the config.** k_max varies per dataset (4 for
  Cora, around 63 for Chameleon), so the module dict is built from the k values actually present.
  Hence `from_config`, which needs the decomposition in hand.
* **Subgraphs sharing a k are batched together.** Rather than looping over every subgraph of every
  graph in Python, all subgraphs with the same k across the whole batch are packed into one
  disjoint union and pushed through that branch once. For graph classification with ~3 subgraphs
  per molecule and a batch of 32, that is 3 backbone calls instead of 96.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from cacose.nn.attention import CrossSubgraphAttention
from cacose.nn.backbone import BACKBONES
from cacose.nn.merge import merge_node_features
from cacose.nn.pooling import POOLINGS
from cacose.types import GraphSample

__all__ = ["CaCoSE"]

SHARED = "shared"


class CaCoSE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        ks: Sequence[int],
        *,
        task: str = "nc",
        hidden: int = 128,
        d_s: int = 128,
        backbone: str = "gcn",
        pooling: str = "sagpool",
        readout: str = "mean+max",
        num_heads: int = 2,
        pool_ratio: float = 0.5,
        share_weights: bool = False,
        attn_residual: bool = False,
        gcn_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if task not in ("nc", "gc"):
            raise ValueError(f"task must be 'nc' or 'gc', got {task!r}")
        if len(ks) == 0:
            raise ValueError("the decomposition produced no subgraphs; nothing to build")

        self.task, self.hidden, self.d_s = task, hidden, d_s
        self.share_weights = share_weights
        self.ks = sorted({int(k) for k in ks})

        branch_keys = [SHARED] if share_weights else [str(k) for k in self.ks]
        self.backbones = nn.ModuleDict(
            {
                key: BACKBONES.create(
                    backbone, in_dim=in_dim, hidden=hidden, num_layers=gcn_layers, dropout=dropout
                )
                for key in branch_keys
            }
        )
        self.poolers = nn.ModuleDict(
            {
                key: POOLINGS.create(
                    pooling, hidden=hidden, d_s=d_s, ratio=pool_ratio, readout=readout
                )
                for key in branch_keys
            }
        )

        self.attention = CrossSubgraphAttention(d_s, num_heads=num_heads, residual=attn_residual)

        head_in = (hidden + d_s) if task == "nc" else d_s
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    @classmethod
    def from_config(cls, cfg, in_dim: int, num_classes: int, ks: Sequence[int]) -> CaCoSE:
        """Build from a resolved config plus the facts only the data can supply."""
        m = cfg.model
        return cls(
            in_dim=in_dim,
            num_classes=num_classes,
            ks=ks,
            task=cfg.task,
            hidden=m.hidden,
            d_s=m.d_s,
            backbone=m.backbone,
            pooling=m.pooling,
            readout=m.readout,
            num_heads=m.num_heads,
            pool_ratio=m.pool_ratio,
            share_weights=m.share_weights,
            attn_residual=m.attn_residual,
            gcn_layers=m.gcn_layers,
            dropout=m.dropout,
        )

    def _branch(self, module_dict: nn.ModuleDict, k: int) -> nn.Module:
        key = SHARED if self.share_weights else str(k)
        if key not in module_dict:
            raise KeyError(
                f"no branch for k={k}; the model was built for ks={self.ks}. "
                "For graph classification the module dict must span the union of k "
                "across the whole dataset, not one graph."
            )
        return module_dict[key]

    def _embed_subgraphs(self, samples: Sequence[GraphSample]):
        """Run every subgraph through its branch, batching all subgraphs that share a k.

        Returns per-graph lists of node-embedding blocks, pooled vectors, and k values.
        """
        by_k: dict[int, list[tuple[int, int]]] = {}
        for gi, sample in enumerate(samples):
            for si, sg in enumerate(sample.decomposition.subgraphs):
                by_k.setdefault(sg.k, []).append((gi, si))

        counts = [len(s.decomposition.subgraphs) for s in samples]
        h_out: list[list[Tensor | None]] = [[None] * n for n in counts]
        z_out: list[list[Tensor | None]] = [[None] * n for n in counts]
        k_out: list[list[int]] = [[sg.k for sg in s.decomposition.subgraphs] for s in samples]

        for k, members in by_k.items():
            xs, eis, batch_vec, sizes = [], [], [], []
            offset = 0
            for slot, (gi, si) in enumerate(members):
                sg = samples[gi].decomposition.subgraphs[si]
                device = samples[gi].x.device
                nmap = sg.node_map.to(device)
                xs.append(samples[gi].x[nmap])
                eis.append(sg.edge_index.to(device) + offset)
                n = int(nmap.numel())
                batch_vec.append(torch.full((n,), slot, dtype=torch.long, device=device))
                sizes.append(n)
                offset += n

            x_cat = torch.cat(xs, dim=0)
            ei_cat = torch.cat(eis, dim=1)
            b_cat = torch.cat(batch_vec)

            h_cat = self._branch(self.backbones, k)(x_cat, ei_cat)
            z_cat = self._branch(self.poolers, k)(h_cat, ei_cat, b_cat)

            start = 0
            for slot, (gi, si) in enumerate(members):
                n = sizes[slot]
                h_out[gi][si] = h_cat[start : start + n]
                z_out[gi][si] = z_cat[slot]
                start += n

        return h_out, z_out, k_out

    def forward(self, samples: GraphSample | Sequence[GraphSample]) -> tuple[Tensor, dict]:
        if isinstance(samples, GraphSample):
            samples = [samples]
        if self.task == "nc" and len(samples) != 1:
            raise ValueError(f"node classification expects one graph, got {len(samples)}")

        h_per_graph, z_per_graph, k_per_graph = self._embed_subgraphs(samples)
        device = samples[0].x.device
        batch = len(samples)

        widths = [len(z) for z in z_per_graph]
        n_max = max(widths)
        z_s = torch.zeros(batch, n_max, self.d_s, device=device)
        pad_mask = torch.ones(batch, n_max, dtype=torch.bool, device=device)
        sub_ids = torch.full((batch, n_max), -1, dtype=torch.long)
        for gi, zs in enumerate(z_per_graph):
            z_s[gi, : len(zs)] = torch.stack(zs)
            pad_mask[gi, : len(zs)] = False
            sub_ids[gi, : len(zs)] = torch.tensor(k_per_graph[gi], dtype=torch.long)

        # only pass a mask when the batch is actually ragged; an all-False mask is wasted work
        mask = pad_mask if bool(pad_mask.any()) else None
        z_attn, attn_weights = self.attention(z_s, key_padding_mask=mask)

        aux = {
            "Z_S": z_s,
            "Z_S_attn": z_attn,
            "attn_weights": attn_weights,
            "subgraph_ids": sub_ids,
        }

        if self.task == "nc":
            z_v = merge_node_features(
                h_per_graph[0], z_attn[0, : widths[0]], samples[0].decomposition
            )
            return self.head(z_v), aux

        # graph readout: mean over real subgraphs only, so padding cannot dilute it
        keep = (~pad_mask).to(z_attn.dtype).unsqueeze(-1)
        z_g = (z_attn * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        aux["Z_G"] = z_g
        return self.head(z_g), aux

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
