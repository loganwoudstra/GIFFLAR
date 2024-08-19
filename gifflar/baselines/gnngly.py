from typing import Dict, Literal
from collections import OrderedDict

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from gifflar.model import DownstreamGGIN

"""
# Model properties extracted from GNNGLY
Learning Rate:    0
Dropout:          0.1
Batch Size:     256
Max Epoch:      200
Num GNN layers:   5
Patience:         0.1
in_feats:       133   [max degree is 12?!]
hidden_dim:      14
"""


class GNNGLY(DownstreamGGIN):
    """
    Implementation follows the loose description in [LINK] as good as possible. Where the specification is unclear, we
    fill in what sounds reasonable
    """

    def __init__(self, hidden_dim: int, num_layers: int, output_dim: int,
                 task: Literal["classification", "regression", "multilabel"], **kwargs):
        """
        Initialize the model following the papers description.
        """
        super().__init__(hidden_dim, output_dim, task)

        del self.convs

        self.layers = nn.Sequential(OrderedDict([(f"layer{l + 1}", GCNConv((133 if l == 0 else hidden_dim), hidden_dim)) for l in range(num_layers)]))

        # ASSUMPTION: mean pooling
        self.pooling = global_mean_pool

    def forward(self, batch):
        """
        Forward the data though the model.

        Args:
            batch: Batch of heterogeneous graphs.

        Returns:
            Dict holding the node embeddings, the graph embedding, and the final model prediction
        """
        # Extract atomic graph from the heterogeneous graph
        x = batch["gnngly_x"].float()
        batch_ids = batch["gnngly_batch"]
        edge_index = batch["gnngly_edge_index"]

        # Propagate the data through the model
        for layer in self.layers:
            x = layer(x, edge_index)

        # Compute the graph embeddings and make the final prediction based on this
        graph_embed = self.pooling(x, batch_ids)
        pred = self.head(graph_embed)

        return {
            "node_embed": x,
            "graph_embed": graph_embed,
            "preds": pred,
        }
