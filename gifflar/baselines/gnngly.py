from typing import Dict

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

    def __init__(self, output_dim, task, **kwargs):
        """
        Initialize the model following the papers description.
        """
        super().__init__(14, output_dim, task)

        # Define the encoders (sizes based on table 1)
        self.atom_encoder = torch.eye(101)
        self.chiral_encoder = torch.eye(4)
        self.degree_encoder = torch.eye(13)
        self.charge_encoder = torch.eye(5)
        self.h_encoder = torch.eye(5)
        self.hybrid_encoder = torch.eye(5)

        # Five layers of plain graph convolution with a hidden dimension of 14.
        self.layers = [
            GCNConv(133, 14),
            GCNConv(14, 14),
            GCNConv(14, 14),
            GCNConv(14, 14),
            GCNConv(14, 14),
        ]

        # ASSUMPTION: mean pooling
        self.pooling = global_mean_pool

        # ASSUMPTION: a prediction head that seems quite elaborate given the other parts of the paper
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(14, 64),
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim),
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """
        Forward the data though the model.

        Args:
            batch: Batch of heterogeneous graphs.

        Returns:
            Dict holding the node embeddings, the graph embedding, and the final model prediction
        """
        # Extract atomic graph from the heterogeneous graph
        x = batch["gnngly_x"]
        batch_ids = batch["gnngly_batch"]
        edge_index = batch["gnngly_edge_index"]

        # Compute the atom-wise encodings
        x = torch.stack([torch.concat([
            self.atom_encoder[a[0]],
            self.chiral_encoder[a[1]],
            self.degree_encoder[a[2]],
            self.charge_encoder[a[3]],
            self.h_encoder[a[4]],
            self.hybrid_encoder[a[5]],
        ]) for a in x])

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
