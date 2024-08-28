from typing import Literal

from torch import nn
from torch_geometric.nn import GINConv

from gifflar.pretransforms import RandomWalkPE, LaplacianPE
from gifflar.utils import get_metrics


def dict_embeddings(dim: int, keys: list[object]):
    emb = nn.Embedding(len(keys) + 1, dim)
    mapping = {key: i for i, key in enumerate(keys)}
    return lambda x: emb[mapping.get(x, len(keys))]


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


def get_prediction_head(input_dim: int, num_predictions: int,
                        task: Literal["regression", "classification", "multilabel"], metric_prefix: str = "") -> tuple:
    head = nn.Sequential(
        nn.Linear(input_dim, input_dim // 2),
        nn.PReLU(),
        nn.Dropout(0.2),
        nn.Linear(input_dim // 2, num_predictions)
    )
    if task == "classification" and num_predictions > 1:
        head.append(nn.Softmax(dim=-1))

    if task == "regression":
        loss = nn.MSELoss()
    elif num_predictions == 1 or task == "multilabel":
        loss = nn.BCEWithLogitsLoss()
    else:
        loss = nn.CrossEntropyLoss()

    metrics = get_metrics(task, num_predictions, metric_prefix)
    return head, loss, metrics


pre_transforms = {
    "LaplacianPE": LaplacianPE,
    "RandomWalkPE": RandomWalkPE,
}


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

    def forward(self, input_, name):
        """
        Embed the input using the specified embedding

        Args:
            input_: The input to the embedding
            name: The name of the embedding to use
        """
        return getattr(self, name).forward(input_)