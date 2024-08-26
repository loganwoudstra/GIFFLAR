from typing import Literal, Any
from collections import OrderedDict

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, GraphConv
from glycowork.glycan_data.loader import lib

from gifflar.data import HeteroDataBatch
from gifflar.model import DownstreamGGIN


class SweetNetLightning(DownstreamGGIN):
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int,
                 task: Literal["classification", "regression", "multilabel"], **kwargs: Any):
        """
        Embed the SweetNet Model into the pytorch-lightning framework.

        Args:
            hidden_dim: Number of hidden dimensions to use in model.
            output_dim: Number of outputs to produce, usually number of classes/labels/tasks.
            num_layers: Number of graph convolutional layers to use in the model.
            task: What kind of dataset the model is trained on, necessary to select the metrics.
            **kwargs: Additional arguments to pass to the model.
        """
        super().__init__(hidden_dim, output_dim, task, **kwargs)

        del self.convs
        del self.head

        # Load the untrained model from glycowork
        self.item_embedding = nn.Embedding(len(lib), hidden_dim)
        self.layers = nn.Sequential(OrderedDict([(f"layer{l + 1}", GraphConv(hidden_dim, hidden_dim)) for l in range(num_layers)]))
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, batch: HeteroDataBatch) -> dict[str, torch.Tensor]:
        """
        Forward the data though the model.

        Args:
            batch: Batch of heterogeneous graphs.

        Returns:
            Dict holding the node embeddings, the graph embedding, and the final model prediction
        """
        # Extract monosaccharide graph from the heterogeneous graph
        x = batch["sweetnet_x"]
        batch_ids = batch["sweetnet_batch"]
        edge_index = batch["sweetnet_edge_index"]

        # Getting node features
        x = self.item_embedding(x)
        x = x.squeeze(1)

        for layer in self.layers:
            x = layer(x, edge_index)

        graph_embed = global_mean_pool(x, batch_ids)
        pred = self.head(graph_embed)

        return {
            "node_embed": x,
            "graph_embed": graph_embed,
            "preds": pred,
        }
