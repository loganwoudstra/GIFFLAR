import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from gifflar.model import DownstreamGGIN

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
    def __init__(self, output_dim, task, **kwargs):
        super().__init__(14, output_dim, task)

        self.atom_encoder = torch.eye(101)
        self.chiral_encoder = torch.eye(4)
        self.degree_encoder = torch.eye(13)
        self.charge_encoder = torch.eye(5)
        self.h_encoder = torch.eye(5)
        self.hybrid_encoder = torch.eye(5)

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

    def to(self, device):
        super(GNNGLY, self).to(device)
        self.layers = [l.to(device) for l in self.layers]
        self.atom_encoder.to(device)
        self.chiral_encoder.to(device)
        self.degree_encoder.to(device)
        self.charge_encoder.to(device)
        self.h_encoder.to(device)
        self.hybrid_encoder.to(device)

    def forward(self, batch):
        x = batch["gnngly_x"]
        batch_ids = batch["gnngly_batch"]
        edge_index = batch["gnngly_edge_index"]

        x = torch.stack([torch.concat([
            self.atom_encoder[a[0]],
            self.chiral_encoder[a[1]],
            self.degree_encoder[a[2]],
            self.charge_encoder[a[3]],
            self.h_encoder[a[4]],
            self.hybrid_encoder[a[5]],
        ]) for a in x]).to(x.device)

        for layer in self.layers:
            x = layer(x, edge_index)

        graph_embed = self.pooling(x, batch_ids)
        pred = self.head(graph_embed)
        return {
            "node_embed": x,
            "graph_embed": graph_embed,
            "preds": pred,
        }
