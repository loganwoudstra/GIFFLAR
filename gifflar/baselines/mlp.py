from gifflar.model import DownstreamGGIN


class MLP(DownstreamGGIN):
    def forward(self, batch):
        pred = self.head(batch["fp"])
        return {
            "node_embed": None,
            "graph_embed": batch["fp"],
            "preds": pred,
        }
