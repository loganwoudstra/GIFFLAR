from glycowork.ml.models import prep_model
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from ssn.model import DownstreamGGIN


class SweetNetLightning(DownstreamGGIN):
    def __init__(self, hidden_dim: int, output_dim: int, task, **kwargs):
        super().__init__(hidden_dim, output_dim, task)

        self.model = prep_model("SweetNet", output_dim, hidden_dim=hidden_dim)

    def forward(self, batch):
        x = batch.x_dict["atoms"]
        batch_ids = batch.batch_dict["atoms"]
        edge_index = batch.edge_index_dict["atoms", "coboundary", "atoms"]

        # Getting node features
        x = self.model.item_embedding(x)
        x = x.squeeze(1)

        # Graph convolution operations
        x = F.leaky_relu(self.model.conv1(x, edge_index))
        x = F.leaky_relu(self.model.conv2(x, edge_index))
        node_embeds = F.leaky_relu(self.model.conv3(x, edge_index))
        graph_embed = global_mean_pool(node_embeds, batch_ids)

        # Fully connected part
        x = self.model.act1(self.model.bn1(self.model.lin1(graph_embed)))
        x_out = self.model.bn2(self.model.lin2(x))
        x = F.dropout(self.model.act2(x_out), p=0.5, training=self.model.training)

        x = self.model.lin3(x).squeeze(1)

        return {
            "node_embed": node_embeds,
            "graph_embed": graph_embed,
            "preds": x,
        }
