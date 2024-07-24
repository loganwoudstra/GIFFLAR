import urllib.request
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from glycowork.glycan_data.loader import df_species as taxonomy
from glyles import convert
from tqdm import tqdm


def iupac2smiles(iupac):
    if any([x in iupac for x in ["?", "{", "}"]]):
        return None
    try:
        smiles = convert(iupac)[0][1]
        if len(smiles) < 10:
            return None
        return smiles
    except:
        return None


def get_taxonomy():
    """
    Download full taxonomy data, process it, and save it as a tsv file.
    """
    if not (p := Path("taxonomy.tsv")).exists():
        mask = []
        for i in tqdm(taxonomy["glycan"]):
            smiles = iupac2smiles(i)
            mask.append(smiles is not None and len(smiles) > 10)
        tax = taxonomy[mask]
        tax.to_csv(p, sep="\t", index=False)
    return pd.read_csv(p, sep="\t")


def bitwise_or_agg(arrays):
    return np.bitwise_or.reduce(arrays)


def get_taxonomic_level(level):
    """
    Extract taxonomy data at a specific level, process it, and save it as a tsv file.
    """
    if not (p := Path(f"taxonomy_{level}.tsv")).exists():
        tax = get_taxonomy()[["glycan", level]]
        tax.rename(columns={"glycan": "IUPAC"}, inplace=True)
        tax[tax[level] == "undetermined"] = np.nan
        tax.dropna(inplace=True)

        tax = pd.concat([tax["IUPAC"], pd.get_dummies(tax[level])], axis=1)
        tax = tax.groupby('IUPAC').agg("sum").reset_index()

        classes = [x for x in tax.columns if x != "IUPAC"]
        tax[classes] = tax[classes].applymap(lambda x: min(1, x))

        tax["split"] = np.random.choice(["train", "val", "test"], tax.shape[0], p=[0.7, 0.2, 0.1])
        # tax.drop(level, axis=1, inplace=True)
        tax.to_csv(p, sep="\t", index=False)
    return p


def get_immunogenicity():
    """
    Download immunogenicity data, process it, and save it as a tsv file.

    Config:
        - name: Immunogenicity
          task: class-1
          label: label

    Returns:
        The filepath of the processed immunogenicity data.
    """
    if not (p := Path("immunogenicity.tsv")).exists():
        urllib.request.urlretrieve(
            "https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_immunogenicity.csv",
            "immunogenicity.csv"
        )
        df = pd.read_csv("immunogenicity.csv")[["glycan", "immunogenicity"]]
        df.rename(columns={"glycan": "IUPAC"}, inplace=True)
        df.dropna(inplace=True)

        classes = {n: i for i, n in enumerate(df["immunogenicity"].unique())}
        df["label"] = df["immunogenicity"].map(classes)
        df["split"] = np.random.choice(["train", "val", "test"], df.shape[0], p=[0.7, 0.2, 0.1])

        df.drop("immunogenicity", axis=1, inplace=True)
        df.to_csv(p, sep="\t", index=False)
        with open("immunogenicity_classes.tsv", "w") as f:
            for n, i in classes.items():
                print(n, i, sep="\t", file=f)
    return p


def get_glycosylation():
    """
    Download glycosylation data, process it, and save it as a tsv file.

    Config:
        - name: Glycosylation
          task: class-1
          label: label

    Returns:
        The filepath of the processed glycosylation data.
    """
    if not (p := Path("glycosylation.tsv")).exists():
        urllib.request.urlretrieve(
            "https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_properties.csv",
            "glycosylation.csv"
        )
        df = pd.read_csv("glycosylation.csv")[["glycan", "link"]]
        df.rename(columns={"glycan": "IUPAC"}, inplace=True)
        df.dropna(inplace=True)

        classes = {n: i for i, n in enumerate(df["link"].unique())}
        df["label"] = df["link"].map(classes)
        df["split"] = np.random.choice(["train", "val", "test"], df.shape[0], p=[0.7, 0.2, 0.1])

        df.drop("link", axis=1, inplace=True)
        df.to_csv(p, sep="\t", index=False)
        with open("glycosylation_classes.tsv", "w") as f:
            for n, i in classes.items():
                print(n, i, sep="\t", file=f)
    return p


def get_dataset(data_config) -> Dict:
    name_fracs = data_config["name"].split("_")
    match name_fracs[0]:
        case "Taxonomy":
            path = get_taxonomic_level(name_fracs[1])
        case "Immunogenicity":
            path = get_immunogenicity()
        case "Glycosylation":
            path = get_glycosylation()
        case "class-1" | "class-n" | "multilabel" | "reg-1" | "reg-n":  # Used for testing
            root = Path("dummy_data")
            if not root.is_dir():
                root = "tests" / root
            path = root / f"{name_fracs[0].replace('-', '_')}.csv"
        case _:  # Unknown dataset
            raise ValueError(f"Unknown dataset {data_config['name']}.")
    # if "label" in data_config:
    #     if data_config["task"] in {"regression", "multilabel"}:
    #         data_config["num_classes"] = len(data_config["label"])
    #     else:
    #         data_config["num_classes"] = int(pd.read_csv(
    #             path, sep="\t" if path.suffix.lower().endswith(".tsv") else ","
    #         )[data_config["label"]].values.max() + 1)
    #         if data_config["num_classes"] == 2:
    #             data_config["num_classes"] = 1
    data_config["filepath"] = path
    return data_config
