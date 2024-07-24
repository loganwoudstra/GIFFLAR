from typing import Dict

import torch

from gifflar.model import DownstreamGGIN


class MLP(DownstreamGGIN):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
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
