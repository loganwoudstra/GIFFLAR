from pathlib import Path
import sys

import pandas as pd
import yaml

for metric in ["Accuracy", "MatthewsCorrCoef", "AUROC", "Sensitivity"]:
    root = Path(sys.argv[1])
    models = [str(x).split("/")[-1] for x in root.iterdir() if x.is_dir()]  # ["rf", "svm", "xgb", "mlp", "gnngly", "sweetnet", "gifflar"]
    print(models)
    datasets = ["Immunogenicity", "Glycosylation", "Domain", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    df = pd.DataFrame(index=datasets, columns=models)
    df.fillna(-1, inplace=True)
    for model in models:
        base = root / model
        for v in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda x: int(x.name.split("_")[1]), reverse=True):
            if not (v / "hparams.yaml").exists() or not (v / "metrics.csv").exists():
                continue
            with open(v / "hparams.yaml", "r") as file:
                name = yaml.load(file, Loader=yaml.FullLoader)["dataset"]["name"]
                if "_" in name:
                    name = name.split("_")[1]
            if df.at[name, model] == -1:
                res = pd.read_csv(v / "metrics.csv", index_col=0)
                col = list(filter(lambda x: "val/" in x and metric in x, res.columns))
                df.at[name, model] = float(res[col].dropna().max())
    df.to_csv(f"results_{metric.lower()}.csv")
