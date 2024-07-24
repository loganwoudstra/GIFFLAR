from typing import List, Literal, Dict, Optional

from glycowork.glycan_data.loader import lib
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch_geometric.nn import GINConv, HeteroConv, global_mean_pool

from gifflar.pretransforms import RandomWalkPE, LaplacianPE
from gifflar.utils import atom_map, bond_map, get_metrics


def dict_embeddings(dim: int, keys: List[object]):
    emb = torch.nn.Embedding(len(keys) + 1, dim)
    mapping = {key: i for i, key in enumerate(keys)}
    return lambda x: emb[mapping.get(x, len(keys))]


def get_gin_layer(hidden_dim: int):
    return GINConv(
        nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_dim),
        )
    )


pre_transforms = {
    "LaplacianPE": LaplacianPE,
    "RandomWalkPE": RandomWalkPE,
}


class GlycanGIN(LightningModule):
    def __init__(self, hidden_dim: int, num_layers: int, task: Literal["regression", "classification", "multilabel"],
                 pre_transform_args: Optional[Dict] = None):
        super().__init__()
        rand_dim = hidden_dim
        self.addendum = []
        if pre_transform_args is not None:
            for name, args in pre_transform_args.items():
                self.addendum.append(pre_transforms[name].attr_name)
                rand_dim -= args["dim"]

        self.embedding = {
            "atoms": nn.Embedding(len(atom_map) + 2, rand_dim),
            "bonds": nn.Embedding(len(bond_map) + 2, rand_dim),
            "monosacchs": nn.Embedding(len(lib) + 2, rand_dim),
        }

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HeteroConv({
                key: get_gin_layer(hidden_dim) for key in [
                    ("atoms", "coboundary", "atoms"),
                    ("atoms", "to", "bonds"),
                    ("bonds", "to", "monosacchs"),
                    ("bonds", "boundary", "bonds"),
                    ("monosacchs", "boundary", "monosacchs")
                ]
            }))

        self.pooling = global_mean_pool

        self.task = task

    def forward(self, batch):
        for key in batch.x_dict.keys():
            # Compute random encodings for the atom type and include positional encodings
            pes = [self.embedding[key](batch.x_dict[key])]
            for pe in self.addendum:
                pes.append(batch[f"{key}_{pe}"])

            batch.x_dict[key] = torch.concat(pes, dim=1)

        for conv in self.convs:
            batch.x_dict = conv(batch.x_dict, batch.edge_index_dict)

        return batch.x_dict, self.pooling(
            torch.concat([batch.x_dict["atoms"], batch.x_dict["bonds"], batch.x_dict["monosacchs"]], dim=0),
            torch.concat([batch.batch_dict["atoms"], batch.batch_dict["bonds"], batch.batch_dict["monosacchs"]], dim=0)
        )


class PretrainGGIN(GlycanGIN):
    pass


class DownstreamGGIN(GlycanGIN):
    def __init__(self, hidden_dim: int, output_dim: int, task: Literal["regression", "classification", "multilabel"],
                 num_layers: int = 3, batch_size: int = 32, pre_transform_args: Optional[Dict] = None, **kwargs):
        super().__init__(hidden_dim, num_layers, task, pre_transform_args)
        self.output_dim = output_dim

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, self.output_dim)
        )
        if self.task == "multilabel":
            self.sigmoid = nn.Sigmoid()

        self.batch_size = batch_size

        if task == "regression":
            self.loss = nn.MSELoss()
        elif self.output_dim == 1:
            self.loss = nn.BCEWithLogitsLoss()
        elif self.task == "multilabel":
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.metrics = get_metrics(self.task, self.output_dim)

    def to(self, device):
        super(DownstreamGGIN, self).to(device)
        for split, metric in self.metrics.items():
            self.metrics[split] = metric.to(device)
        self.embedding = {k: e.to(device) for k, e in self.embedding.items()}

    def forward(self, batch):
        node_embed, graph_embed = super().forward(batch)
        pred = self.head(graph_embed)
        return {
            "node_embed": node_embed,
            "graph_embed": graph_embed,
            "preds": pred,
        }

    def shared_step(self, batch, stage: str):
        fwd_dict = self.forward(batch)
        fwd_dict["labels"] = batch["y_oh"] if self.task == "multilabel" else batch["y"]

        if self.task != "multilabel":
            if list(fwd_dict["preds"].shape) == [len(batch["y"]), 1]:
                fwd_dict["preds"] = fwd_dict["preds"][:, 0]
        else:
            fwd_dict["preds"] = self.sigmoid(fwd_dict["preds"])

        if self.output_dim == 1 or self.task == "multilabel":
            fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape).float())
        elif self.task == "classification":
            fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape[:-1]))
        else:
            fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"])

        if self.task == "classification" and self.output_dim > 1:
            self.metrics[stage].update(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape[:-1]))
        else:
            self.metrics[stage].update(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape))

        self.log(f"{stage}/loss", fwd_dict["loss"], batch_size=self.batch_size)
        return fwd_dict

    def training_step(self, batch):
        return self.shared_step(batch, "train")

    def validation_step(self, batch):
        return self.shared_step(batch, "val")

    def test_step(self, batch):
        return self.shared_step(batch, "test")

    def shared_end(self, stage: str):
        metrics = self.metrics[stage].compute()
        self.log_dict(metrics)
        self.metrics[stage].reset()

    def on_train_epoch_end(self) -> None:
        self.shared_end("train")

    def on_validation_epoch_end(self) -> None:
        self.shared_end("val")

    def on_test_epoch_end(self) -> None:
        self.shared_end("test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5),
            "monitor": "val/loss",
        }
