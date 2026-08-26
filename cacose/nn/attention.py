"""Cross-subgraph attention.

Pooling collapses each subgraph independently, which throws away how the regions relate. This
is where that is recovered: standard scaled dot-product self-attention over the *stack* of
subgraph embeddings, so every subgraph can attend to every other regardless of hop distance.
That is the paper's mechanism for restoring long-range signal after decomposition.

Two details are load-bearing:

* `average_attn_weights=False` -- per-head weights are kept, because the Phase 3 visualiser
  reads them and averaging destroys what it wants to show.
* residual + LayerNorm is off by default. The paper states the attention plainly, without
  either (spec ambiguity #5), so the flag exists to test the alternative rather than to
  silently improve on the paper.
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["CrossSubgraphAttention"]


class CrossSubgraphAttention(nn.Module):
    def __init__(
        self,
        d_s: int,
        num_heads: int = 1,
        residual: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_s % num_heads != 0:
            raise ValueError(f"d_s={d_s} must be divisible by num_heads={num_heads}")
        self.d_s, self.num_heads, self.residual = d_s, num_heads, residual
        self.mha = nn.MultiheadAttention(d_s, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_s) if residual else None

    def forward(
        self, z_s: Tensor, key_padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """`z_s` is [B, N_S, d_s]; returns (attended [B, N_S, d_s], weights [B, H, N_S, N_S]).

        `key_padding_mask` is [B, N_S] with True at padded positions -- needed for graph
        classification, where each graph in a batch has its own number of subgraphs.
        """
        if z_s.dim() != 3:
            raise ValueError(f"expected [B, N_S, d_s], got {tuple(z_s.shape)}")
        out, weights = self.mha(
            z_s,
            z_s,
            z_s,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        if self.norm is not None:
            out = self.norm(out + z_s)
        return out, weights

    def extra_repr(self) -> str:
        return f"d_s={self.d_s}, num_heads={self.num_heads}, residual={self.residual}"
