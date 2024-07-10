import shutil
from pathlib import Path

import yaml

from gifflar.train import main


def test_dummy_datasets():
    # Initial clean-up
    if (data := Path("data")).exists():
        shutil.rmtree(data)
    if (logs := Path("logs")).exists():
        shutil.rmtree(logs)

    main(Path("dummy_data") / "datasets.yaml")

    for i, name in enumerate(["class_1", "class_n", "multilabel", "reg_1", "reg_n"]):
        # Check that the preprocessing files are created
        for split in ["train", "val", "test"]:
            assert (data / name / f"{split}.pt").exists()
        for step in ["filter", "transform"]:
            assert (data / name / "processed" / f"pre_{step}.pt").exists()

        # Check that the result files are created
        for model in ["rf", "svm", "xgb", "mlp", "gnngly", "sweetnet", "ssn"]:
            assert (logs / model / f"version_{i}" / "metrics.csv").exists()
            assert (hp := (logs / model / f"version_{i}" / "hparams.yaml")).exists()
            with open(hp, "r") as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            assert config["dataset"]["name"] == name.replace("_", "-")

    shutil.rmtree(data)
    shutil.rmtree("logs")
