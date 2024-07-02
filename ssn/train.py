import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import CSVLogger
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData

from ssn.data import DownsteamGDM
from ssn.benchmarks import get_taxonomic_level
from ssn.model import DownstreamGGIN


def check_data(d: HeteroData):
    if d is None or \
            not torch.gt(torch.tensor(d["atoms"].x.shape[0]), torch.tensor(10)) or \
            not torch.gt(torch.tensor(d["bonds"].x.shape[0]), torch.tensor(10)) or \
            not torch.gt(torch.tensor(d["atoms", "coboundary", "atoms"].edge_index.shape[1]), torch.tensor(1)) or \
            not torch.gt(torch.tensor(d["atoms", "to", "bonds"].edge_index.shape[1]), torch.tensor(1)) or \
            not torch.gt(torch.tensor(d["bonds", "to", "monosacchs"].edge_index.shape[1]), torch.tensor(1)) or \
            not torch.gt(torch.tensor(d["bonds", "boundary", "bonds"].edge_index.shape[1]), torch.tensor(1)) or \
            not torch.ge(torch.tensor(d["monosacchs"].x.shape[0]), torch.tensor(1)) or \
            not torch.ge(torch.tensor(d["monosacchs", "boundary", "monosacchs"].edge_index.shape[1]), torch.tensor(0)):
            # not torch.gt(torch.tensor(d["bonds", "coboundary", "bonds"].edge_index.shape[1]), torch.tensor(1)) or \
        return False
    return True


def train(batch_size: int, seed: int, taxonomy_level: str = "Domain"):
    seed_everything(seed)
    path = get_taxonomic_level(taxonomy_level)
    datamodule = DownsteamGDM(path, batch_size)
    model = DownstreamGGIN(hidden_dim=128, output_dim=5, num_layers=3, batch_size=batch_size)
    logger = CSVLogger("logs", name="ssn")
    trainer = Trainer(
        callbacks=[
            # ModelCheckpoint(save_last=True, mode="min", monitor="val/reg/loss", save_top_k=1),
            RichModelSummary(),
            RichProgressBar(),
        ],
        max_epochs=100,
        logger=logger
    )
    trainer.fit(model, datamodule)


# train(32, 42, "Kingdom")
train(32, 42, "Subdomain")
