from typing import List

from glycowork.glycan_data.loader import lib
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch_geometric.nn import GINConv, HeteroConv, global_mean_pool
from torchmetrics import Accuracy, MetricCollection, MatthewsCorrCoef, AUROC

from ssn.utils import atom_map, bond_map


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


class GlycanGIN(LightningModule):
    def __init__(self, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.embedding = {
            "atoms": nn.Embedding(len(atom_map) + 1, hidden_dim),
            "bonds": nn.Embedding(len(bond_map) + 1, hidden_dim),
            "monosacchs": nn.Embedding(len(lib) + 1, hidden_dim),
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

    def forward(self, x_dict, edge_index_dict, batch_dict):
        # for node_type, x in x_dict.items():
        #     x_dict[node_type] = self.embedding[node_type](x)

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict, self.pooling(
            torch.concat([x_dict["atoms"], x_dict["bonds"], x_dict["monosacchs"]], dim=0),
            torch.concat([batch_dict["atoms"], batch_dict["bonds"], batch_dict["monosacchs"]], dim=0)
        )


class DownstreamGGIN(GlycanGIN):
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int = 3, batch_size: int = 32, **kwargs):
        super().__init__(hidden_dim, num_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.batch_size = batch_size

        if output_dim == 1:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.metrics = self._metrics(output_dim)

    def _metrics(self, num_classes: int):
        if num_classes == 1:
            m = MetricCollection([
                Accuracy(task="binary"),
                AUROC(task="binary"),
                MatthewsCorrCoef(task="binary"),
            ])
        else:
            m = MetricCollection([
                Accuracy(task="multiclass", num_classes=num_classes),
                AUROC(task="multiclass", num_classes=num_classes),
                MatthewsCorrCoef(task="multiclass", num_classes=num_classes),
            ])
        return {"train": m.clone(prefix="train/"), "val": m.clone(prefix="val/"), "test": m.clone(prefix="test/")}

    def forward(self, x_dict, edge_index_dict, batch_dict):
        node_embed, graph_embed = super().forward(x_dict, edge_index_dict, batch_dict)
        pred = self.head(graph_embed)
        return {
            "node_embed": node_embed,
            "graph_embed": graph_embed,
            "preds": pred,
        }

    def shared_step(self, batch, batch_idx, stage: str):
        fwd_dict = self.forward(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        fwd_dict["labels"] = batch["y"]
        fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"])
        self.metrics[stage].update(fwd_dict["preds"], fwd_dict["labels"])
        self.log(f"{stage}/loss", fwd_dict["loss"], batch_size=self.batch_size)
        return fwd_dict

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")

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
