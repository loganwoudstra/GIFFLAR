from ssn.model import DownstreamGGIN


class MLP(DownstreamGGIN):
    def forward(self, batch):
        return {
            "node_embed": None,
            "graph_embed": batch["fp"],
            "preds": self.head(batch["fp"]),
        }
