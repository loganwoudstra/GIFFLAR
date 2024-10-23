import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from argparse import ArgumentParser
import time

from numpy.f2py.cfuncs import callbacks
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import CSVLogger
from sympy.physics.units import acceleration
from torch_geometric import seed_everything

from experiments.lgi_model import LGI_Model
from gifflar.data.modules import DownsteamGDM, LGI_GDM
from gifflar.model.base import GlycanGIN
from gifflar.model.baselines.sweetnet import SweetNetLightning
from gifflar.pretransforms import get_pretransforms
from gifflar.train import setup
from gifflar.utils import read_yaml_config, hash_dict


GLYCAN_ENCODERS = {
    "gifflar": GlycanGIN,
    "sweetnet": SweetNetLightning,
}


def main(config):
    kwargs = read_yaml_config(config)
    kwargs["pre-transforms"] = {"GIFFLARTransform": "", "SweetNetTransform": ""}
    kwargs["hash"] = hash_dict(kwargs["pre-transforms"])
    seed_everything(kwargs["seed"])

    datamodule = LGI_GDM(
        root=kwargs["root_dir"], filename=kwargs["origin"], hash_code=kwargs["hash"],
        batch_size=kwargs["model"].get("batch_size", 1), transform=None,
        pre_transform=get_pretransforms("", **(kwargs["pre-transforms"] or {})),
    )

    # set up the logger
    glycan_model_name = kwargs["model"]["glycan_encoder"]["name"] + (kwargs["model"]["glycan_encoder"].get("suffix", None) or "")
    lectin_model_name = kwargs["model"]["lectin_encoder"]["name"] + (kwargs["model"]["lectin_encoder"].get("suffix", None) or "")
    logger = CSVLogger(kwargs["logs_dir"], name="LGI_" + glycan_model_name + lectin_model_name)
    logger.log_hyperparams(kwargs)

    glycan_encoder = GLYCAN_ENCODERS[kwargs["model"]["glycan_encoder"]["name"]](**kwargs["model"]["glycan_encoder"])
    model = LGI_Model(
        glycan_encoder,
        kwargs["model"]["lectin_encoder"]["name"],
        kwargs["model"]["lectin_encoder"]["layer_num"],
    )
    model.to("cuda")
    
    trainer = Trainer(
        callbacks=[RichProgressBar(), RichModelSummary()],
        logger=logger,
        max_epochs=kwargs["model"]["epochs"],
        accelerator="gpu",
    )
    start = time.time()
    trainer.fit(model, datamodule)
    print("Training took", time.time() - start, "s")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    main(parser.parse_args().config)
