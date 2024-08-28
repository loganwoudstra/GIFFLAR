from typing import Any

import torch

from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.downstream import DownstreamGGIN


class MLP(DownstreamGGIN):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        del self.convs

    def forward(self, batch: HeteroDataBatch) -> dict[str, torch.Tensor]:
        """
        Make predictions based on the molecular fingerprint.

        Args:
            batch: Batch of heterogeneous graphs.

        Returns:
            Dict holding the node embeddings (None for the MLP), the graph embedding, and the final model prediction
        """
        return {
            "node_embed": None,
            "graph_embed": batch["fp"],
            "preds": self.head(batch["fp"]),
        }
