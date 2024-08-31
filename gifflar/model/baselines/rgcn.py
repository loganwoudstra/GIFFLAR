from typing import Literal, Optional, Any

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GINConv

from gifflar.model.downstream import DownstreamGGIN


class HeteroPReLU(nn.Module):
    def __init__(self, prelus: dict[str, nn.PReLU]):
        """
        A module that applies a PReLU activation function to each input.

        Args:
            prelus: The PReLU activations to apply to each input.
        """
        super(HeteroPReLU, self).__init__()
        for name, prelu in prelus.items():
            setattr(self, name, prelu)

    def forward(self, input_: dict) -> dict:
        """
        Apply the PReLU activation to the input.

        Args:
            input_: The input to apply the activation to.

        Returns:
            The input with the PReLU activation applied.
        """
        for key, value in input_.items():
            input_[key] = getattr(self, key).forward(value)
        return input_


class RGCN(DownstreamGGIN):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int,
            output_dim: int,
            task: Literal["regression", "classification", "multilabel"],
            num_layers: int = 3,
            batch_size: int = 32,
            pre_transform_args: Optional[dict] = None,
            **kwargs: Any
    ):
        """
        Implementation of the relational GCN model.

        Args:
            feat_dim: The feature dimension of the model.
            hidden_dim: The dimensionality of the hidden layers.
            output_dim: The output dimension of the model
            task: The type of task to perform.
            num_layers: The number of layers in the network.
            batch_size: The batch size to use
            pre_transform_args: The arguments for the pre-transforms.
            kwargs: Additional arguments
        """
        super(RGCN, self).__init__(feat_dim, hidden_dim, output_dim, task, num_layers, batch_size, pre_transform_args,
                                   **kwargs)

        self.convs = torch.nn.ModuleList()
        dims = [kwargs["feat_dim"], hidden_dim // 2] + [hidden_dim] * (num_layers - 1)
        for i in range(num_layers):
            convs = {
                # Set the inner layers to be a single weight without using the nodes embedding (therefore, e=-1)
                key: GINConv(nn.Sequential(nn.Linear(dims[i], dims[i + 1])), eps=-1) for key in [
                    ("atoms", "coboundary", "atoms"),
                    ("atoms", "to", "bonds"),
                    ("bonds", "to", "monosacchs"),
                    ("bonds", "boundary", "bonds"),
                    ("monosacchs", "boundary", "monosacchs")
                ]
            }
            self_loop_weight = nn.Sequential(nn.Linear(dims[i], dims[i + 1]))
            convs.update({
                ("atoms", "self", "atoms"): GINConv(self_loop_weight, eps=-1),
                ("bonds", "self", "bonds"): GINConv(self_loop_weight, eps=-1),
                ("monosacchs", "self", "monosacchs"): GINConv(self_loop_weight, eps=-1),
            })
            self.convs.append(HeteroConv(convs))
            self.convs.append(HeteroPReLU({
                "atoms": nn.PReLU(),
                "bonds": nn.PReLU(),
                "monosacchs": nn.PReLU(),
            }))
