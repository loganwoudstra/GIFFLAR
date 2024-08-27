from typing import List, Literal, Dict, Optional, Any

from glycowork.glycan_data.loader import lib
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINConv, HeteroConv, global_mean_pool

from gifflar.loss import MultiLoss
from gifflar.pretransforms import RandomWalkPE, LaplacianPE
from gifflar.utils import atom_map, bond_map, get_metrics, mono_map


def dict_embeddings(dim: int, keys: List[object]):
    emb = torch.nn.Embedding(len(keys) + 1, dim)
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


class GlycanGIN(LightningModule):
    def __init__(self, feat_dim: int, hidden_dim: int, num_layers: int, batch_size: int = 32,
                 pre_transform_args: Optional[Dict] = None):
        """
        Initialize the GlycanGIN model, the base for all DL-models in this package

        Args:
            feat_dim: The feature dimension of the model
            hidden_dim: The hidden dimension of the model
            num_layers: The number of GIN layers to use
            pre_transform_args: A dictionary of pre-transforms to apply to the input data
        """
        super().__init__()

        # Preparation for additional information in the embeddings of the nodes
        rand_dim = feat_dim
        self.addendum = []
        if pre_transform_args is not None:
            for name, args in pre_transform_args.items():
                if name in pre_transforms:
                    self.addendum.append(pre_transforms[name].attr_name)
                    rand_dim -= args["dim"]

        # Set up the learnable embeddings for all node types
        self.embedding = MultiEmbedding({
            "atoms": nn.Embedding(len(atom_map) + 2, rand_dim),
            "bonds": nn.Embedding(len(bond_map) + 2, rand_dim),
            "monosacchs": nn.Embedding(len(lib) + 2, rand_dim),
        })

        # Define the GIN layers to embed messages between nodes
        self.convs = torch.nn.ModuleList()
        dims = [feat_dim]
        if feat_dim <= hidden_dim // 2:
            dims += [hidden_dim // 2]
        else:
            dims += [hidden_dim]
        dims += [hidden_dim] * (num_layers - 1)
        for i in range(num_layers):
            self.convs.append(HeteroConv({
                key: get_gin_layer(dims[i], dims[i + 1]) for key in [
                    ("atoms", "coboundary", "atoms"),
                    ("atoms", "to", "bonds"),
                    ("bonds", "to", "monosacchs"),
                    ("bonds", "boundary", "bonds"),
                    ("monosacchs", "boundary", "monosacchs")
                ]
            }))

        self.batch_size = batch_size

    def forward(self, batch: HeteroData) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            batch: The batch of data to process

        Returns:
            node_embed: The node embeddings
            graph_embed: The graph embeddings
        """
        for key in batch.x_dict.keys():
            # Compute random encodings for the atom type and include positional encodings
            pes = [self.embedding.forward(batch.x_dict[key], key)]
            for pe in self.addendum:
                pes.append(batch[f"{key}_{pe}"])

            batch.x_dict[key] = torch.concat(pes, dim=1)

        for conv in self.convs:
            if isinstance(conv, HeteroConv):
                batch.x_dict = conv(batch.x_dict, batch.edge_index_dict)
            else:  # the layer is an activation function from the RGCN
                batch.x_dict = conv(batch.x_dict)

        return batch.x_dict

    def shared_step(self, batch: HeteroData, stage: Literal["train", "val", "test"]) -> dict:
        raise NotImplementedError()

    def training_step(self, batch: HeteroData, batch_idx: int) -> dict:
        """Compute the training step of the model"""
        return self.shared_step(batch, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int) -> dict:
        """Compute the validation step of the model"""
        return self.shared_step(batch, "val")

    def test_step(self, batch: HeteroData, batch_idx: int) -> dict:
        """Compute the testing step of the model"""
        return self.shared_step(batch, "test")

    def shared_end(self, stage: Literal["train", "val", "test"]) -> None:
        raise NotImplementedError()

    def on_train_epoch_end(self) -> None:
        """Compute the end of the training epoch"""
        self.shared_end("train")

    def on_validation_epoch_end(self) -> None:
        """Compute the end of the validation"""
        self.shared_end("val")

    def on_test_epoch_end(self) -> None:
        """Compute the end of the testing"""
        self.shared_end("test")

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler of the model"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5),
            "monitor": "val/loss",
        }


class PretrainGGIN(GlycanGIN):
    def __init__(self, hidden_dim: int, tasks: list[dict[str, Any]], num_layers: int = 3, batch_size: int = 32,
                 pre_transform_args: Optional[Dict] = None, **kwargs):
        """
        Initialize the PretrainGGIN model, a pre-training model for downstream tasks.

        Args:
            hidden_dim: The hidden dimension of the model
            output_dim: The output dimension of the model
            num_layers: The number of GIN layers to use
            batch_size: The batch size to use
            pre_transform_args: A dictionary of pre-transforms to apply to the input data
            kwargs: Additional arguments
        """
        super().__init__(kwargs["feat_dim"], hidden_dim, num_layers, batch_size, pre_transform_args)
        self.tasks = tasks

        # Define the prediction heads of the model
        # for t, task in enumerate(tasks):
        #     model, loss, metrics = get_prediction_head(hidden_dim, task["num_classes"], task["task"])
        self.atom_mask_head, self.atom_mask_loss, self.atom_mask_metrics \
            = get_prediction_head(hidden_dim, len(atom_map) + 1, "classification", "atom")
        self.bond_mask_head, self.bond_mask_loss, self.bond_mask_metrics \
            = get_prediction_head(hidden_dim, len(bond_map) + 1, "classification", "bond")
        self.mono_pred_head, self.mono_pred_loss, self.mono_pred_metrics \
            = get_prediction_head(hidden_dim, len(mono_map) + 1, "classification", "mono")
        self.mods_pred_head, self.mods_pred_loss, self.mods_pred_metrics \
            = get_prediction_head(hidden_dim, 16, "multilabel", "mods")

        self.loss = MultiLoss(4, dynamic=kwargs.get("loss", "static") == "dynamic")

    def to(self, device):
        self.atom_mask_head.to(device)
        self.atom_mask_loss.to(device)
        self.atom_mask_metrics["train"].to(device)
        self.atom_mask_metrics["val"].to(device)
        self.atom_mask_metrics["test"].to(device)

        self.bond_mask_head.to(device)
        self.bond_mask_loss.to(device)
        self.bond_mask_metrics["train"].to(device)
        self.bond_mask_metrics["val"].to(device)
        self.bond_mask_metrics["test"].to(device)

        self.mono_pred_head.to(device)
        self.mono_pred_loss.to(device)
        self.mono_pred_metrics["train"].to(device)
        self.mono_pred_metrics["val"].to(device)
        self.mono_pred_metrics["test"].to(device)

        self.mods_pred_head.to(device)
        self.mods_pred_loss.to(device)
        self.mods_pred_metrics["train"].to(device)
        self.mods_pred_metrics["val"].to(device)
        self.mods_pred_metrics["test"].to(device)

        return super().to(device)

    def forward(self, batch: HeteroData) -> dict:
        """
        Forward pass of the model.

        Args:
            batch: The batch of data to process

        Returns:
            node_embed: The node embeddings
        """
        node_embeds = super().forward(batch)
        atom_preds = self.atom_mask_head(node_embeds["atoms"])
        bond_preds = self.bond_mask_head(node_embeds["bonds"])
        mono_preds = self.mono_pred_head(node_embeds["monosacchs"])
        mods_preds = self.mods_pred_head(node_embeds["monosacchs"])
        return {
            "node_embed": node_embeds,
            "atom_preds": atom_preds,
            "bond_preds": bond_preds,
            "mono_preds": mono_preds,
            "mods_preds": mods_preds,
        }

    def shared_step(self, batch: HeteroData, stage: Literal["train", "val", "test"]) -> dict:
        fwd_dict = self.forward(batch)

        # print(self.mono_pred_head)
        # print(type(self.mono_pred_loss))
        # print(fwd_dict["mono_preds"])
        # print(batch["bonds_y"])
        # print(batch["mono_y"])
        # print(torch.tensor(batch["mono_y"]).reshape(fwd_dict["mono_preds"].shape[:-1]))
        # print(torch.tensor(batch["mono_y"]).reshape(fwd_dict["mono_preds"].shape[:-1]).shape)

        fwd_dict["atom_loss"] = self.atom_mask_loss(fwd_dict["atom_preds"], batch["atoms_y"] - 1)
        fwd_dict["bond_loss"] = self.bond_mask_loss(fwd_dict["bond_preds"], batch["bonds_y"] - 1)
        fwd_dict["mono_loss"] = self.mono_pred_loss(fwd_dict["mono_preds"], torch.tensor(batch["mono_y"]).reshape(fwd_dict["mono_preds"].shape[:-1]))
        fwd_dict["mods_loss"] = self.mods_pred_loss(fwd_dict["mods_preds"], batch["mods_y"].float())
        fwd_dict["loss"] = self.loss([
            fwd_dict["atom_loss"], fwd_dict["bond_loss"], fwd_dict["mono_loss"], fwd_dict["mods_loss"]
        ])

        self.atom_mask_metrics[stage].update(fwd_dict["atom_preds"], batch["atoms_y"] - 1)
        self.bond_mask_metrics[stage].update(fwd_dict["bond_preds"], batch["bonds_y"] - 1)
        self.mono_pred_metrics[stage].update(fwd_dict["mono_preds"], torch.tensor(batch["mono_y"]).reshape(fwd_dict["mono_preds"].shape[:-1]))
        self.mods_pred_metrics[stage].update(fwd_dict["mods_preds"], batch["mods_y"])

        self.log(f"{stage}/atom_loss", fwd_dict["atom_loss"], batch_size=self.batch_size)
        self.log(f"{stage}/bond_loss", fwd_dict["bond_loss"], batch_size=self.batch_size)
        self.log(f"{stage}/mono_loss", fwd_dict["mono_loss"], batch_size=self.batch_size)
        self.log(f"{stage}/mods_loss", fwd_dict["mods_loss"], batch_size=self.batch_size)
        self.log(f"{stage}/loss", fwd_dict["loss"], batch_size=self.batch_size)

        return fwd_dict

    def shared_end(self, stage: Literal["train", "val", "test"]) -> None:
        metrics = self.atom_mask_metrics[stage].compute()
        self.log_dict(metrics)
        self.atom_mask_metrics[stage].reset()

        metrics = self.bond_mask_metrics[stage].compute()
        self.log_dict(metrics)
        self.bond_mask_metrics[stage].reset()

        metrics = self.mono_pred_metrics[stage].compute()
        self.log_dict(metrics)
        self.mono_pred_metrics[stage].reset()

        metrics = self.mods_pred_metrics[stage].compute()
        self.log_dict(metrics)
        self.mods_pred_metrics[stage].reset()


class DownstreamGGIN(GlycanGIN):
    def __init__(self, hidden_dim: int, output_dim: int, task: Literal["regression", "classification", "multilabel"],
                 num_layers: int = 3, batch_size: int = 32, pre_transform_args: Optional[Dict] = None, **kwargs):
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

        self.pooling = global_mean_pool
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

    def forward(self, batch):
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
        graph_embed = self.pooling(
            torch.concat([node_embed["atoms"], node_embed["bonds"], node_embed["monosacchs"]], dim=0),
            torch.concat([batch.batch_dict["atoms"], batch.batch_dict["bonds"], batch.batch_dict["monosacchs"]], dim=0)
        )
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
