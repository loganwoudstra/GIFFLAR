import copy
import hashlib
import json

import torch
import yaml
from jsonargparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import CSVLogger
from torch_geometric import seed_everything

from gifflar.baselines.gnngly import GNNGLY
from gifflar.baselines.mlp import MLP
from gifflar.baselines.sweetnet import SweetNetLightning
from gifflar.data import DownsteamGDM
from gifflar.benchmarks import get_dataset
from gifflar.model import DownstreamGGIN
from gifflar.pretransforms import get_pretransforms
from gifflar.utils import get_sl_model, get_metrics

MODELS = {
    "gifflar": DownstreamGGIN,
    "gnngly": GNNGLY,
    "mlp": MLP,
    "sweetnet": SweetNetLightning,
}


def setup(**kwargs):
    seed_everything(kwargs["seed"])
    data_config = get_dataset(kwargs["dataset"])
    datamodule = DownsteamGDM(
        root=kwargs["root_dir"], filename=data_config["filepath"], hash_code=kwargs["hash"],
        batch_size=kwargs["model"].get("batch_size", 1), transform=None,
        pre_transform=get_pretransforms(**(kwargs["pre-transforms"] or {})), **data_config,
    )
    data_config["num_classes"] = datamodule.train.dataset_args["num_classes"]
    logger = CSVLogger(kwargs["logs_dir"], name=kwargs["model"]["name"] + kwargs["model"].get("suffix", ""))
    kwargs["dataset"]["filepath"] = str(data_config["filepath"])
    logger.log_hyperparams(kwargs)
    metrics = get_metrics(data_config["task"], data_config["num_classes"])
    return data_config, datamodule, logger, metrics


def fit(**kwargs):
    data_config, datamodule, logger, metrics = setup(**kwargs)
    model = get_sl_model(kwargs["model"]["name"], data_config["task"], data_config["num_classes"], **kwargs)
    train_X, train_y, train_yoh = datamodule.train.to_statistical_learning()
    model.fit(train_X, train_yoh if data_config["task"] == "multilabel" else train_y)

    for X, y, yoh, name in [
        (train_X, train_y, train_yoh, "train"),
        (*datamodule.val.to_statistical_learning(), "val"),
        (*datamodule.test.to_statistical_learning(), "test"),
    ]:
        labels = torch.tensor(
            yoh if data_config["task"] == "multilabel" else y,
            dtype=torch.long if data_config["task"] != "regression" else torch.float
        )

        preds = torch.tensor(model.predict_proba(X) if data_config["task"] in {"classification", "multilabel"} else
                             model.predict(X), dtype=torch.float)

        if data_config["task"] == "classification":
            if data_config["num_classes"] > 1:
                labels = labels[:, 0]
                if kwargs["model"]["name"] == "xgb":
                    preds = preds[0]
            else:
                preds = preds[:, 1]
                labels = labels.reshape(-1)
        elif data_config["task"] == "multilabel" and len(preds.shape) == 3:
            preds = preds[:, :, 1].T
        elif data_config["num_classes"] == 1:  # data_config["task"] == "regression"
            preds = preds.reshape(labels.shape)
        else:
            preds = preds.reshape(-1)
            labels = labels.reshape(-1)

        metrics[name].update(preds, labels)
        logger.log_metrics(metrics[name].compute())
    logger.save()


def train(**kwargs):
    data_config, datamodule, logger, _ = setup(**kwargs)
    model = MODELS[kwargs["model"]["name"]](output_dim=data_config["num_classes"], task=data_config["task"],
                                            pre_transform_args=kwargs["pre-transforms"], **kwargs["model"])
    trainer = Trainer(
        callbacks=[
            # ModelCheckpoint(save_last=True, mode="min", monitor="val/reg/loss", save_top_k=1),
            RichModelSummary(),
            RichProgressBar(),
        ],
        # accelerator="cpu",
        max_epochs=kwargs["model"]["epochs"],
        logger=logger,
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
    if isinstance(config["datasets"], dict):
        datasets = [config["datasets"]]
    else:
        datasets = config["datasets"]
    del config["datasets"]

    if isinstance(config["model"], dict):
        models = [config["model"]]
    else:
        models = config["model"]
    del config["model"]

    for dataset in datasets:
        for model in models:
            tmp_config = copy.deepcopy(config)
            tmp_config["dataset"] = dataset
            if "label" in tmp_config["dataset"] and not isinstance(tmp_config["dataset"]["label"], list):
                tmp_config["dataset"]["label"] = [tmp_config["dataset"]["label"]]
            tmp_config["model"] = model
            yield tmp_config


def hash_dict(input_dict, n_chars: int = 8):
    # Convert the dictionary to a JSON string
    dict_str = json.dumps(input_dict, sort_keys=True)

    # Generate a SHA-256 hash of the string
    hash_obj = hashlib.sha256(dict_str.encode())

    # Get the first 8 characters of the hexadecimal digest
    hash_str = hash_obj.hexdigest()[:n_chars]

    return hash_str


def main(config):
    custom_args = read_yaml_config(config)
    for args in unfold_config(custom_args):
        # try:
        args["hash"] = hash_dict(args["pre-transforms"])
        print(args)
        if args["model"]["name"] in ["rf", "svm", "xgb"]:
            fit(**args)
        else:
            # continue
            train(**args)
        print("Finished", args["model"]["name"], "on", args["dataset"]["name"])
        # except Exception as e:
        #     print(args["model"]["name"], "failed on", args["dataset"]["name"], "with", f"\"{e}\"")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    main(parser.parse_args().config)
