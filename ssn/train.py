import torch
import yaml
from jsonargparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import CSVLogger
from torch_geometric import seed_everything

from ssn.baselines.gnngly import GNNGLY
from ssn.baselines.mlp import MLP
from ssn.baselines.sweetnet import SweetNetLightning
from ssn.data import DownsteamGDM
from ssn.benchmarks import get_dataset
from ssn.model import DownstreamGGIN

torch.autograd.set_detect_anomaly(True)

models = {
    "ssn": DownstreamGGIN,
    "gnngly": GNNGLY,
    "mlp": MLP,
    "sweetnet": SweetNetLightning,
}


def train(**kwargs):
    seed_everything(kwargs["seed"])
    data_config = get_dataset(kwargs["dataset-name"])
    datamodule = DownsteamGDM(root=kwargs["root_dir"], filename=data_config["filepath"],
                              batch_size=kwargs["model"]["batch_size"], **kwargs)
    model = models[kwargs["model"]["name"]](output_dim=data_config["num_classes"], **kwargs["model"])
    logger = CSVLogger("logs", name=kwargs["model"]["name"])
    trainer = Trainer(
        callbacks=[
            # ModelCheckpoint(save_last=True, mode="min", monitor="val/reg/loss", save_top_k=1),
            RichModelSummary(),
            RichProgressBar(),
        ],
        max_epochs=kwargs["model"]["epochs"],
        logger=logger
    )
    trainer.fit(model, datamodule)


def read_yaml_config(filename: str) -> dict:
    """Read in yaml config for training."""
    with open(filename, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def merge_dicts(a: dict, b: dict):
    out = a
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            out[key] = merge_dicts(a[key], b[key])
        else:
            out[key] = b[key]
    return out


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    default_args = read_yaml_config("configs/default.yaml")
    custom_args = read_yaml_config(parser.parse_args().config)
    merged_args = merge_dicts(default_args, custom_args)
    print(merged_args)
    train(**merged_args)
