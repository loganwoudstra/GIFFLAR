from typing import Literal, Any

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool

from gifflar.utils import get_metrics


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
            input_dim: The input dimension of the pooling layer
            output_dim: The output dimension of the pooling layer
        """
        super().__init__()
        match (mode):
            case "global_mean":
                self.pooling = global_mean_pool
            case _:
                raise ValueError(f"Pooling mode {mode} not implemented yet.")

    def forward(self, nodes: dict[str, torch.Tensor], batch_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform pooling on the input

        Args:
            x: The input to the pooling layer
        """
        return self.pooling(
            torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
            torch.concat([batch_ids["atoms"], batch_ids["bonds"], batch_ids["monosacchs"]], dim=0)
        )
