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
from ssn.utils import get_sl_model, get_metrics

models = {
    "ssn": DownstreamGGIN,
    "gnngly": GNNGLY,
    "mlp": MLP,
    "sweetnet": SweetNetLightning,
}


def fit(**kwargs):
    seed_everything(kwargs["seed"])
    data_config = get_dataset(kwargs["dataset-name"])
    datamodule = DownsteamGDM(root=kwargs["root_dir"], filename=data_config["filepath"], batch_size=1, **kwargs)
    logger = CSVLogger("logs", name=kwargs["model"]["name"])
    model = get_sl_model(kwargs["model"]["name"], data_config["task"], data_config["num_classes"], **kwargs)
    metrics = get_metrics(data_config["task"], data_config["num_classes"])

    train_X, train_y = datamodule.train.to_statistical_learning()
    model.fit(train_X, train_y)

    for X, y, name in [
        (train_X, train_y, "train"),
        (*datamodule.val.to_statistical_learning(), "val"),
        (*datamodule.test.to_statistical_learning(), "test"),
    ]:
        preds = model.predict_proba(X)
        t_preds = torch.tensor(preds, dtype=torch.float)
        t_labels = torch.tensor(y, dtype=torch.long)
        metrics[name].update(t_preds, t_labels)
        logger.log_metrics(metrics[name].compute())
    logger.save()


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
    # default_args = read_yaml_config("configs/default.yaml")
    custom_args = read_yaml_config(parser.parse_args().config)
    # merged_args = merge_dicts(default_args, custom_args)
    # print(merged_args)
    if custom_args["model"]["name"] in ["rf", "svm", "xgb"]:
        fit(**custom_args)
    else:
        train(**custom_args)
