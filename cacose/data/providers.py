"""Concrete dataset providers.

Phase 1 needs three. A new application domain is a fourth class registered here; nothing else
in the package changes.
"""

from __future__ import annotations

import torch
from torch_geometric.datasets import Planetoid, TUDataset, WikipediaNetwork

from cacose.data.base import PROVIDERS, DatasetProvider, GraphBundle, normalize_graph

__all__ = ["PlanetoidProvider", "WikipediaNetworkProvider", "TUDatasetProvider"]


@PROVIDERS.register("planetoid")
class PlanetoidProvider(DatasetProvider):
    """Citation networks: Cora, CiteSeer, PubMed. Homophilic node classification."""

    NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}

    def available(self) -> list[str]:
        return sorted(self.NAMES)

    def load(self, name: str) -> GraphBundle:
        ds = Planetoid(root=str(self.root / "planetoid"), name=self.NAMES[name.lower()])
        data = normalize_graph(ds[0])
        return GraphBundle(
            name=name.lower(),
            task="nc",
            graphs=[data],
            num_features=ds.num_features,
            num_classes=ds.num_classes,
            labels=data.y,
        )


@PROVIDERS.register("wikipedia")
class WikipediaNetworkProvider(DatasetProvider):
    """Chameleon and Squirrel -- the heterophilic datasets.

    `geom_gcn_preprocess=True` selects the Geom-GCN version, which is what the paper's baseline
    numbers correspond to (not the later Platonov et al. filtered release). It also ships 10
    split columns; those are 60/20/20 while the paper states 48/32/20, so they are carried in
    `provided_masks` for comparison but are not the default (spec ambiguity #7).
    """

    NAMES = {"chameleon", "squirrel"}

    def available(self) -> list[str]:
        return sorted(self.NAMES)

    def load(self, name: str) -> GraphBundle:
        ds = WikipediaNetwork(
            root=str(self.root / "wikipedia"), name=name.lower(), geom_gcn_preprocess=True
        )
        data = normalize_graph(ds[0])
        masks = {
            split: getattr(data, f"{split}_mask")
            for split in ("train", "val", "test")
            if hasattr(data, f"{split}_mask")
        }
        return GraphBundle(
            name=name.lower(),
            task="nc",
            graphs=[data],
            num_features=ds.num_features,
            num_classes=ds.num_classes,
            labels=data.y,
            provided_masks=masks or None,
            meta={"provided_mask_ratio": "60/20/20 (Geom-GCN)"},
        )


@PROVIDERS.register("tudataset")
class TUDatasetProvider(DatasetProvider):
    """Graph classification: MUTAG, PROTEINS, IMDB-*, COLLAB, REDDIT-BINARY."""

    NAMES = {
        "mutag": "MUTAG",
        "proteins": "PROTEINS",
        "imdb-b": "IMDB-BINARY",
        "imdb-m": "IMDB-MULTI",
        "collab": "COLLAB",
        "rdt-b": "REDDIT-BINARY",
    }

    def available(self) -> list[str]:
        return sorted(self.NAMES)

    def load(self, name: str) -> GraphBundle:
        ds = TUDataset(root=str(self.root / "tu"), name=self.NAMES[name.lower()])
        graphs, labels = [], []
        for g in ds:
            g = normalize_graph(g)
            if g.x is None:
                # Featureless social datasets (IMDB, COLLAB, REDDIT). Constant features let the
                # structure carry the signal; a degree one-hot would be the richer alternative.
                g.x = torch.ones(g.num_nodes, 1)
            graphs.append(g)
            labels.append(int(g.y))
        return GraphBundle(
            name=name.lower(),
            task="gc",
            graphs=graphs,
            num_features=graphs[0].x.size(-1),
            num_classes=ds.num_classes,
            labels=torch.tensor(labels, dtype=torch.long),
        )
