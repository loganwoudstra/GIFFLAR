import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.transforms import BaseTransform

from ssn.model import DownstreamGGIN

"""
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
    def __init__(self, output_dim, **kwargs):
        super().__init__(14, output_dim)

        self.layers = [
            GCNConv(133, 14),
            GCNConv(14, 14),
            GCNConv(14, 14),
            GCNConv(14, 14),
            GCNConv(14, 14),
        ]

        self.pooling = global_mean_pool

        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(14, 64),
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim),
        )

    def forward(self, batch):
        x = batch.x_dict["atoms"]
        batch_ids = batch.batch_dict["atoms"]
        edge_index = batch.edge_index_dict["atoms", "coboundary", "atoms"]

        for layer in self.layers:
            x = layer(x, edge_index)

        graph_embed = self.pooling(x, batch_ids)
        pred = self.head(graph_embed)
        return {
            "node_embed": x,
            "graph_embed": graph_embed,
            "preds": pred,
        }
