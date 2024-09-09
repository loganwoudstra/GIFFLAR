from typing import Literal, Any

import torch
from torch import nn, scatter_add
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax
from torch_geometric.nn import GINConv, global_mean_pool

from gifflar.utils import get_metrics


class GlobalAttention(torch.nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(self, gate_nn, nn=None):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)


    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)

        # out = scatter_add(gate * x, index=batch, dim=0)  # , dim_size=size)
        src = torch.zeros_like(size)
        out = src.scatter_add(src=gate * x, index=batch, dim=0)

        return out


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(gate_nn={self.gate_nn}, '
                f'nn={self.nn})')


def get_gin_layer(input_dim: int, output_dim: int) -> GINConv:
    """
    Get a GIN layer with the specified input and output dimensions

    Args:
        input_dim: The input dimension of the GIN layer
        output_dim: The output dimension of the GIN layer

    Returns:
        A GIN layer with the specified input and output dimensions
    """
    return GINConv(
        nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(output_dim),
        )
    )


def get_prediction_head(
        input_dim: int,
        num_predictions: int,
        task: Literal["regression", "classification", "multilabel"],
        metric_prefix: str = ""
) -> tuple[nn.Module, nn.Module, dict[str, nn.Module]]:
    """
    Create the prediction head for the specified dimensions and generate the loss and metrics for the task

    Args:
        input_dim: The input dimension of the prediction head
        num_predictions: The number of predictions to make
        task: The task to perform, either "regression", "classification" or "multilabel"
        metric_prefix: The prefix to use for the metrics

    Returns:
        A tuple containing the prediction head, the loss function and the metrics for the task
    """
    # Create the prediction head
    head = nn.Sequential(
        nn.Linear(input_dim, input_dim // 2),
        nn.PReLU(),
        nn.Dropout(0.2),
        nn.Linear(input_dim // 2, num_predictions)
    )
    # Add a softmax layer if the task is classification and there are multiple predictions
    if task == "classification" and num_predictions > 1:
        head.append(nn.Softmax(dim=-1))

    # Create the loss function based on the task and the number of outputs to predict
    if task == "regression":
        loss = nn.MSELoss()
    elif num_predictions == 1 or task == "multilabel":
        loss = nn.BCEWithLogitsLoss()
    else:
        loss = nn.CrossEntropyLoss()

    # Get the metrics for the task
    metrics = get_metrics(task, num_predictions, metric_prefix)
    return head, loss, metrics


class MultiEmbedding(nn.Module):
    """Class storing multiple embeddings in a dict-format and allowing for training them as nn.Module"""

    def __init__(self, embeddings: dict[str, nn.Embedding]):
        """
        Initialize the MultiEmbedding class

        Args:
            embeddings: A dictionary of embeddings to store
        """
        super().__init__()
        for name, embedding in embeddings.items():
            setattr(self, name, embedding)

    def forward(self, input_: dict[str, Any], name: str) -> torch.Tensor:
        """
        Embed the input using the specified embedding

        Args:
            input_: The input to the embedding
            name: The name of the embedding to use
        """
        return getattr(self, name).forward(input_)


class GIFFLARPooling(nn.Module):
    """Class to perform pooling on the output of the GIFFLAR model"""

    def __init__(self, mode: str = "global_mean"):
        """
        Initialize the GIFFLARPooling class

        Args:
            mode: The pooling mode to use, either
                - global_mean:
                    Standard mean pooling over all nodes in the graph
                - local_mean:
                    Standard mean pooling over all nodes of each type, then mean over the node types
                - weighted_mean:
                    Local mean pooling with learnable weights per node type in the second mean-computation
                - global_attention:
                    Standard self-attention mechanism over all nodes in the graph
                - local_attention:
                    Standard self-attention mechanism over all nodes of each type, then mean over the nodes types
                - weighted_attention:
                    Standard self-attention mechanism over all nodes of each type, then mean with learnable weights
                        per node type in the cell aggregation.
                - cell_attention:
                    Local mean pooling with self-attention over the cell results
                - local_cell_attention:
                    Local mean pooling with self-attention over the cell results of each type, then mean over the cell types
                - weighted_cell_attention:
                    Local mean pooling with self-attention over the cell results (their means) and learnable weights for aggregation
        """
        super().__init__()
        self.mode = mode
        self.attention = None
        self.weights = None
        if "weighted" in mode:
            self.weights = nn.Parameter(torch.ones(3), requires_grad=True)
        if "attention" in mode:
            if mode == "global_attention":
                self.attention = {"": GlobalAttention(
                    gate_nn=get_gin_layer(1024, 1),
                    nn=get_gin_layer(1024, 1024),
                )}
            else:
                self.attention = {key: GlobalAttention(
                    gate_nn=get_gin_layer(1024, 1),
                    nn=get_gin_layer(1024, 1024),
                ) for key in ["atoms", "bonds", "monosacchs"]}

    def forward(self, nodes: dict[str, torch.Tensor], batch_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform pooling on the input

        Args:
            x: The input to the pooling layer
        """
        match self.mode:
            case "local_mean":
                nodes = {key: global_mean_pool(nodes[key], batch_ids[key]) for key in nodes.keys()}
            case "weighted_mean":
                nodes = {key: global_mean_pool(nodes[key], batch_ids[key]) * self.weights[i] for i, key in enumerate(["atoms", "bonds", "monosacchs"])}
            case "global_attention":
                nodes = {key: self.attention[""](nodes[key], batch_ids[key]) for key in nodes.keys()}
            case "local_attention":
                nodes = {key: self.attention[key](nodes[key], batch_ids[key]) for key in nodes.keys()}
            case "weighted_attention":
                nodes = {key: self.attention[key](nodes[key], batch_ids[key]) * self.weights[i] for i, key in enumerate(["atoms", "bonds", "monosacchs"])}
            case "cell_attention":
                nodes = {key: self.attention[""](global_mean_pool(nodes[key], batch_ids[key])) for key in nodes.keys()}
            case "local_cell_attention":
                nodes = {key: self.attention[key](global_mean_pool(nodes[key], batch_ids[key])) for key in nodes.keys()}
            case "weighted_cell_attention":
                nodes = {key: self.attention[key](global_mean_pool(nodes[key], batch_ids[key])) * self.weights[i] for i, key in enumerate(["atoms", "bonds", "monosacchs"])}
        return global_mean_pool(
            torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
            torch.concat([batch_ids["atoms"], batch_ids["bonds"], batch_ids["monosacchs"]], dim=0)
        )
