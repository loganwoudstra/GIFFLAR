from ssn.model import DownstreamGGIN


class MLP(DownstreamGGIN):
    def forward(self, batch):
        pred = self.head(batch["fp"])
        if self.task != "multilabel" and list(pred.shape) == [len(batch["y"]), 1]:
            pred = pred[:, 0]
        return {
            "node_embed": None,
            "graph_embed": batch["fp"],
            "preds": pred,
        }
