import copy

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

MODELS = {
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
    logger.log_hyperparams(kwargs)
    model = get_sl_model(kwargs["model"]["name"], data_config["task"], data_config["num_classes"], **kwargs)
    metrics = get_metrics(data_config["task"], data_config["num_classes"])

    train_X, train_y, train_yoh = datamodule.train.to_statistical_learning()
    if data_config["task"] == "classification" and data_config["num_classes"] > 2:
        model.fit(train_X, train_yoh)
    else:
        model.fit(train_X, train_y)

    for X, y, yoh, name in [
        (train_X, train_y, train_yoh, "train"),
        (*datamodule.val.to_statistical_learning(), "val"),
        (*datamodule.test.to_statistical_learning(), "test"),
    ]:
        if kwargs["model"]["name"] == "svm":
            preds = model.predict(X)
        else:
            preds = model.predict_proba(X)

        if data_config["task"] == "classification" and data_config["num_classes"] == 1 and len(preds.shape) >= 2:
            preds = preds[:, 1]
        if list(torch.tensor(preds).shape) == [data_config["num_classes"], len(X), 2]:
            preds = torch.tensor(preds)[:, :, 1].T

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
    model = MODELS[kwargs["model"]["name"]](output_dim=data_config["num_classes"], **kwargs["model"])
    logger = CSVLogger("logs", name=kwargs["model"]["name"])
    logger.log_hyperparams(kwargs)
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


def unfold_config(config):
    if isinstance(config["dataset-name"], str):
        dataset_names = [config["dataset-name"]]
    else:
        dataset_names = config["dataset-name"]
    del config["dataset-name"]

    if isinstance(config["model"], dict):
        models = [config["model"]]
    else:
        models = config["model"]
    del config["model"]

    for dataset_name in dataset_names:
        for model in models:
            tmp_config = copy.deepcopy(config)
            tmp_config["dataset-name"] = dataset_name
            tmp_config["model"] = model
            yield tmp_config


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    custom_args = read_yaml_config(parser.parse_args().config)
    for args in unfold_config(custom_args):
        try:
            if args["model"]["name"] in ["rf", "svm", "xgb"]:
                fit(**args)
            else:
                train(**args)
        except Exception as e:
            print(f"Error: {e}")
