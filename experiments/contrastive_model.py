from typing import Any, Literal

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import HeteroData

from experiments.lgi_model import LGI_Model
from experiments.protein_encoding import EMBED_SIZES
from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.base import GlycanGIN
from gifflar.model.baselines.sweetnet import SweetNetLightning


def sigmoid_cosine_distance_p(x, y, p=1):
    sig = torch.nn.Sigmoid()
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - sig(cosine_sim(x, y))) ** p


def cosine_distance_p(x, y, p=1):
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - cosine_sim(x, y)) ** p


DISTANCES = {
    "sigmoid": sigmoid_cosine_distance_p,
    "cosine": cosine_distance_p,
}


class ContrastLGIModel(LGI_Model):
    def __init__(
            self,
            glycan_encoder: GlycanGIN | SweetNetLightning,
            lectin_encoder: str,
            le_layer_num: int,
            latent_dim: int = 256,
            margin_distance: Literal["sigmoid", "cosine"] = "cosine",
            margin: float = 0.25,
            ** kwargs: Any,
    ):
        super(ContrastLGIModel, self).__init__(glycan_encoder, lectin_encoder, le_layer_num, **kwargs)

        self.latent_dim = latent_dim

        self.glycan_red = nn.Linear(glycan_encoder.hidden_dim, latent_dim)
        self.lectin_red = nn.Linear(EMBED_SIZES[lectin_encoder], latent_dim)

        self.pred_loss = nn.MSELoss()
        self.embed_loss = nn.TripletMarginWithDistanceLoss(distance_function=DISTANCES[margin_distance], margin=margin)

    def to(self, device: torch.device):
        super(ContrastLGIModel, self).to(device)
        self.glycan_red.to(device)
        self.lectin_red.to(device)
        return self

    def forward(self, data: HeteroDataBatch, decoys: HeteroDataBatch | None) -> dict[str, torch.Tensor]:
        glycan_embed = self.glycan_encoder(data)["graph_embed"]
        glycan_small = self.glycan_red(glycan_embed)

        lectin_embed = self.lectin_embeddings.batch_query(data["aa_seq"])
        lectin_small = self.lectin_red(lectin_embed)

        fwd_dict = {
            "glycan": glycan_small,
            "lectin": lectin_small,
        }
        if decoys is not None:
            decoy_embed = self.glycan_encoder(decoys)["graph_embed"]
            decoy_small = self.glycan_red(decoy_embed)
            fwd_dict["decoy"] = decoy_small

        return fwd_dict

    def shared_step(self, batch: HeteroDataBatch, decoys: HeteroDataBatch | None, stage: str) -> dict[str, torch.Tensor]:
        fwd_dict = self.forward(batch, decoys)
        fwd_dict["labels"] = batch["y"].float()

        inter_pred = sigmoid_cosine_distance_p(fwd_dict["glycan"], fwd_dict["lectin"])
        fwd_dict["preds"] = inter_pred

        inter_loss = self.pred_loss(inter_pred, fwd_dict["labels"])
        fwd_dict["inter_loss"] = inter_loss.float()

        if decoys is not None:
            embed_loss = self.embed_loss(fwd_dict["glycan"], fwd_dict["lectin"], fwd_dict["decoy"])
            fwd_dict["embed_loss"] = embed_loss
            loss = inter_loss + embed_loss # weighting factor?
        else:
            loss = inter_loss
        fwd_dict["loss"] = loss.float()

        self.metrics[stage].update(inter_pred, fwd_dict["labels"])
        self.log(f"{stage}/loss", fwd_dict["loss"], batch_size=len(fwd_dict["preds"]))
        return fwd_dict

    def training_step(self, batch: tuple[HeteroDataBatch, HeteroDataBatch | None], batch_idx: int) -> dict[str, torch.Tensor]:
        """Compute the training step of the model"""
        return self.shared_step(batch[0], batch[1], "train")

    def validation_step(self, batch: tuple[HeteroDataBatch, HeteroDataBatch | None], batch_idx: int) -> dict[str, torch.Tensor]:
        """Compute the validation step of the model"""
        return self.shared_step(batch[0], batch[1], "val")

    def test_step(self, batch: tuple[HeteroDataBatch, HeteroDataBatch | None], batch_idx: int) -> dict[str, torch.Tensor]:
        """Compute the testing step of the model"""
        return self.shared_step(batch[0], batch[1], "test")

    def predict_step(self, batch: tuple[HeteroDataBatch, HeteroDataBatch | None], batch_idx: int) -> dict[str, torch.Tensor]:
        fwd_dict = self(batch)
        fwd_dict["IUPAC"] = batch["IUPAC"]
        fwd_dict["seq"] = batch["aa_seq"]
        return fwd_dict

    def shared_end(self, stage: Literal["train", "val", "test"]):
        """
        Compute the shared end of the model.

        Args:
            stage: The stage of the model
        """
        metrics = self.metrics[stage].compute()
        self.log_dict(metrics)
        self.metrics[stage].reset()

    def on_train_epoch_end(self) -> None:
        """Compute the end of the training epoch"""
        self.shared_end("train")

    def on_validation_epoch_end(self) -> None:
        """Compute the end of the validation"""
        self.shared_end("val")
        self.lectin_embeddings.close()

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
