from typing import Literal, Optional

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_mean_pool

from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.base import GlycanGIN
from gifflar.model.utils import GIFFLARPooling
from gifflar.utils import get_metrics


class DownstreamGGIN(GlycanGIN):
    def __init__(self, hidden_dim: int, output_dim: int, task: Literal["regression", "classification", "multilabel"],
                 num_layers: int = 3, batch_size: int = 32, pre_transform_args: Optional[dict] = None, **kwargs):
        """
        Initialize the DownstreamGGIN model, a downstream model from a pre-trained GlycanGIN model.

        Args:
            hidden_dim: The hidden dimension of the model
            output_dim: The output dimension of the model
            task: The task to perform, either "regression", "classification" or "multilabel"
            num_layers: The number of GIN layers to use
            batch_size: The batch size to use
            pre_transform_args: A dictionary of pre-transforms to apply to the input data
            kwargs: Additional arguments
        """
        super().__init__(kwargs["feat_dim"], hidden_dim, num_layers, batch_size, pre_transform_args)
        self.output_dim = output_dim

        self.pooling = GIFFLARPooling()
        self.task = task

        # Define the classification head of the model
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, self.output_dim)
        )
        if self.task == "multilabel":
            self.head.append(nn.Sigmoid())

        # Define the loss function based on the task and the number of outputs to predict
        if self.task == "regression":
            self.loss = nn.MSELoss()
        elif self.output_dim == 1:
            self.loss = nn.BCEWithLogitsLoss()
        elif self.task == "multilabel":
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self.metrics = get_metrics(self.task, self.output_dim)

    def to(self, device: torch.device):
        """
        Move the model to the specified device.

        Args:
            device: The device to move the model to

        Returns:
            self: The model moved to the specified device
        """
        super(DownstreamGGIN, self).to(device)
        for split, metric in self.metrics.items():
            self.metrics[split] = metric.to(device)
        return self

    def forward(self, batch: HeteroDataBatch) -> dict:
        """
        Forward pass of the model.

        Args:
            batch: The batch of data to process

        Returns:
            A dictionary containing:
                node_embed: The node embeddings
                graph_embed: The graph embeddings
                preds: The predictions of the model
        """
        node_embed = super().forward(batch)
        graph_embed = self.pooling(node_embed, batch.batch_dict)
        pred = self.head(graph_embed)
        return {
            "node_embed": node_embed,
            "graph_embed": graph_embed,
            "preds": pred,
        }

    def shared_step(self, batch: HeteroData, stage: Literal["train", "val", "test"]) -> dict:
        """
        Shared step for training, validation and testing steps.

        Args:
            batch: The batch of data to process
            stage: The stage of the model, either "train", "val" or "test"

        Returns:
            A dictionary containing:
                node_embed: The node embeddings
                graph_embed: The graph embeddings
                preds: The predictions of the model
                labels: The true labels of the data
                loss: The loss of the model
        """
        fwd_dict = self.forward(batch)
        fwd_dict["labels"] = batch["y_oh"] if self.task == "multilabel" else batch["y"]

        # Adjust the predictions to match the output dimension and eventually apply the sigmoid function
        if self.task != "multilabel":
            if list(fwd_dict["preds"].shape) == [len(batch["y"]), 1]:
                fwd_dict["preds"] = fwd_dict["preds"][:, 0]

        # Compute the loss based on the task and the number of outputs to predict
        if self.output_dim == 1 or self.task == "multilabel":
            fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape).float())
        elif self.task == "classification":
            fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape[:-1]))
        else:
            fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"])

        # Update the metrics based on the task and the number of outputs to predict
        if self.task == "classification" and self.output_dim > 1:
            self.metrics[stage].update(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape[:-1]))
        else:
            self.metrics[stage].update(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape))

        # Log the loss of the model
        self.log(f"{stage}/loss", fwd_dict["loss"], batch_size=self.batch_size)
        return fwd_dict

    def shared_end(self, stage: Literal["train", "val", "test"]) -> None:
        """
        Shared end for training, validation and testing steps.

        Args:
            stage: The stage of the model, either "train", "val" or "test"
        """
        metrics = self.metrics[stage].compute()
        self.log_dict(metrics)
        self.metrics[stage].reset()
