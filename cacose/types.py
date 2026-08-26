"""Core data structures.

`Subgraph` and `Decomposition` are fixed by the design spec (writeups/, section 2.6) because
Phase 2 loads cached decompositions and Phase 3 reads the model's `aux` dict. Changing either
shape means changing both consumers, so they live here rather than inside a module that might
be refactored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from torch import Tensor

__all__ = ["Subgraph", "Decomposition", "Splits", "RunResult", "GraphSample"]


@dataclass(frozen=True)
class Subgraph:
    """One cohesive subgraph: every edge whose coreness score equals `k`.

    `edge_index` is in *local* indexing and carries both directions of each undirected edge,
    so it can be handed to a PyG conv unchanged. `node_map` maps local index -> global node id.
    """

    k: int
    edge_index: Tensor  # [2, 2 * num_undirected_edges], local indices
    node_map: Tensor  # [num_local_nodes], local -> global

    @property
    def num_nodes(self) -> int:
        return int(self.node_map.numel())

    @property
    def num_undirected_edges(self) -> int:
        return int(self.edge_index.size(1)) // 2

    def __repr__(self) -> str:
        return f"Subgraph(k={self.k}, nodes={self.num_nodes}, edges={self.num_undirected_edges})"


@dataclass(frozen=True)
class Decomposition:
    """A partition of the *edge* set into cohesive subgraphs.

    Every edge belongs to exactly one subgraph; a node may appear in several. Note that `S_k`
    is not the k-core `G_k` -- see the spec.
    """

    subgraphs: list[Subgraph]
    num_nodes: int
    kmax: int

    @property
    def ks(self) -> list[int]:
        return [sg.k for sg in self.subgraphs]

    def __len__(self) -> int:
        return len(self.subgraphs)

    def __repr__(self) -> str:
        return f"Decomposition(ks={self.ks}, num_nodes={self.num_nodes}, kmax={self.kmax})"

    def node_membership(self) -> dict[int, list[int]]:
        """global node id -> the k values of the subgraphs containing it."""
        out: dict[int, list[int]] = {}
        for sg in self.subgraphs:
            for g in sg.node_map.tolist():
                out.setdefault(g, []).append(sg.k)
        return out

    def num_isolated_nodes(self) -> int:
        """Nodes in no subgraph. They receive a zero embedding, so they are worth counting."""
        seen = {g for sg in self.subgraphs for g in sg.node_map.tolist()}
        return self.num_nodes - len(seen)


@dataclass(frozen=True)
class Splits:
    """Train/val/test as index tensors.

    Node indices for node classification, graph indices for graph classification -- the same
    container serves both so `Task` is the only place that knows the difference.
    """

    train: Tensor
    val: Tensor
    test: Tensor

    @property
    def sizes(self) -> tuple[int, int, int]:
        return (int(self.train.numel()), int(self.val.numel()), int(self.test.numel()))


@dataclass
class RunResult:
    """One (config, seed) run. Serialised to JSON; field names are consumed by the aggregator."""

    dataset: str
    task: str
    seed: int
    config_hash: str
    best_val_acc: float = 0.0
    test_acc: float = 0.0
    epochs_run: int = 0
    best_epoch: int = 0
    wall_time_s: float = 0.0
    kmax: int = 0
    num_subgraphs: int = 0
    num_isolated_nodes: int = 0
    num_params: int = 0
    git_sha: str = ""
    torch_version: str = ""
    pyg_version: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GraphSample:
    """One graph plus its decomposition -- the unit the model consumes.

    Node classification passes a single sample; graph classification passes a list. Keeping the
    decomposition alongside the features means the model never has to look anything up.
    """

    x: Tensor
    decomposition: Decomposition
    y: Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    def to(self, device) -> GraphSample:
        return GraphSample(
            x=self.x.to(device),
            decomposition=self.decomposition,
            y=None if self.y is None else self.y.to(device),
        )
