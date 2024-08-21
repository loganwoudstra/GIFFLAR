from typing import Literal, Optional, Dict

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GINConv

from gifflar.model import GlycanGIN


class RGCN(GlycanGIN):
    def __init__(self, hidden_dim: int, num_layers: int, task: Literal["regression", "classification", "multilabel"],
                 pre_transform_args: Optional[Dict] = None):
        """
        Implementation of the relational GCN model.

        Args:
            hidden_dim: The dimensionality of the hidden layers.
            num_layers: The number of layers in the network.
            task: The type of task to perform.
            pre_transform_args: The arguments for the pre-transforms.
        """
        super(RGCN, self).__init__(hidden_dim, num_layers, task, pre_transform_args)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HeteroConv({
                # Set the inner layers to be a single weight without using the nodes embedding (therefore, e=-1)
                key: GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim)), eps=-1) for key in [
                    ("atoms", "coboundary", "atoms"),
                    ("atoms", "to", "bonds"),
                    ("bonds", "to", "monosacchs"),
                    ("bonds", "boundary", "bonds"),
                    ("monosacchs", "boundary", "monosacchs")
                ]
            }))
            self.convs.append(nn.PReLU())
