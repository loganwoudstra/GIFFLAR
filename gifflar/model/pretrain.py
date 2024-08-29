import copy
from typing import Literal, Optional, Any

import torch
from torch_geometric.nn import HeteroConv

from gifflar.data.hetero import HeteroDataBatch
from gifflar.loss import MultiLoss
from gifflar.model.base import GlycanGIN
from gifflar.model.utils import get_prediction_head
from gifflar.utils import mono_map, bond_map, atom_map


class PretrainGGIN(GlycanGIN):
    def __init__(self, hidden_dim: int, tasks: list[dict[str, Any]] | None, num_layers: int = 3, batch_size: int = 32,
                 pre_transform_args: Optional[dict] = None, **kwargs):
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

    def forward(self, batch: HeteroDataBatch) -> dict:
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

    def predict_step(self, batch: HeteroDataBatch) -> dict:
        """
        Predict the output of the model and return the node embeddings from all layers

        Args:
            batch: The batch of data to process

        Returns:
            A dictionary containing:
                node_embeds: The node embeddings from all layers
        """

        for key in batch.x_dict.keys():
            # Compute random encodings for the atom type and include positional encodings
            pes = [self.embedding.forward(batch.x_dict[key], key)]
            for pe in self.addendum:
                pes.append(batch[f"{key}_{pe}"])

            batch.x_dict[key] = torch.concat(pes, dim=1)

        node_embeds = []
        for conv in self.convs:
            if isinstance(conv, HeteroConv):
                for key in batch.x_dict:
                    print(key, batch.x_dict[key].shape)
                for key in batch.edge_index_dict:
                    print(key, batch.edge_index_dict[key].shape)
                batch.x_dict = conv(batch.x_dict, batch.edge_index_dict)
                node_embeds.append(copy.deepcopy(batch.x_dict))
            else:  # the layer is an activation function from the RGCN
                batch.x_dict = conv(batch.x_dict)

        return {"node_embeds": node_embeds}

    def shared_step(self, batch: HeteroDataBatch, stage: Literal["train", "val", "test"]) -> dict:
        fwd_dict = self.forward(batch)

        fwd_dict["atom_loss"] = self.atom_mask_loss(fwd_dict["atom_preds"], batch["atoms_y"] - 1)
        fwd_dict["bond_loss"] = self.bond_mask_loss(fwd_dict["bond_preds"], batch["bonds_y"] - 1)
        fwd_dict["mono_loss"] = self.mono_pred_loss(fwd_dict["mono_preds"], torch.tensor(batch["mono_y"]).reshape(
            fwd_dict["mono_preds"].shape[:-1]))
        fwd_dict["mods_loss"] = self.mods_pred_loss(fwd_dict["mods_preds"], batch["mods_y"].float())
        fwd_dict["loss"] = self.loss([
            fwd_dict["atom_loss"], fwd_dict["bond_loss"], fwd_dict["mono_loss"], fwd_dict["mods_loss"]
        ])

        self.atom_mask_metrics[stage].update(fwd_dict["atom_preds"], batch["atoms_y"] - 1)
        self.bond_mask_metrics[stage].update(fwd_dict["bond_preds"], batch["bonds_y"] - 1)
        self.mono_pred_metrics[stage].update(fwd_dict["mono_preds"],
                                             torch.tensor(batch["mono_y"]).reshape(fwd_dict["mono_preds"].shape[:-1]))
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
