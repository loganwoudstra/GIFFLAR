from typing import Dict, Literal

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool, Sequential

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
        super().__init__(14, output_dim, task)

        del self.convs
        del self.head

        self.layers = Sequential('x, edge_index', [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])

        # Five layers of plain graph convolution with a hidden dimension of 14.
        # self.layers = [
        #     GCNConv(133, 14),
        #     GCNConv(14, 14),
        #     GCNConv(14, 14),
        #     GCNConv(14, 14),
        #     GCNConv(14, 14),
        # ]

        # ASSUMPTION: mean pooling
        self.pooling = global_mean_pool

        # ASSUMPTION: a prediction head that seems quite elaborate given the other parts of the paper
        # self.head = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(14, 64),
        #     nn.PReLU(),
        #     nn.BatchNorm1d(64),
        #     nn.Linear(64, output_dim),
        # )

    def to(self, device):
        super(GNNGLY, self).to(device)
        # self.layers = [l.to(device) for l in self.layers]


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
        # for layer in self.layers:
        #     x = layer(x, edge_index)

        # Compute the graph embeddings and make the final prediction based on this
        node_embed = self.layers(x, edge_index)
        graph_embed = self.pooling(node_embed, batch_ids)
        pred = self.head(graph_embed)

        return {
            "node_embed": x,
            "graph_embed": graph_embed,
            "preds": pred,
        }
