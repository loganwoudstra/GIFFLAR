import urllib.request
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from glycowork.glycan_data.loader import df_species as taxonomy
from glyles import convert
from tqdm import tqdm

from ssn.utils import DatasetInfo, RawDataInfo


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
    if not (p := Path("taxonomy.tsv")).exists():
        mask = []
        for i in tqdm(taxonomy["glycan"]):
            mask.append(iupac2smiles(i) is not None)
        tax = taxonomy[mask]
        tax.to_csv(p, sep="\t", index=False)
    return pd.read_csv(p, sep="\t")


def get_taxonomic_level(level):
    if not (p := Path(f"taxonomy_{level}.tsv")).exists():
        tax = get_taxonomy()[["glycan", level]]
        tax = tax[~tax[level].isna()]
        tax["label"] = list(np.array(pd.get_dummies(tax[level]), dtype=int))
        tax = tax.sample(frac=1)
        tax["split"] = np.random.choice(["train", "val", "test"], tax.shape[0], p=[0.7, 0.2, 0.1])
        tax.to_csv(p, sep="\t", index=False)
    return p


def get_immunogenicity():
    if not (p := Path("immunogenicity.tsv")).exists():
        urllib.request.urlretrieve(
            "https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_immunogenicity.csv",
            "immunogenicity.csv"
        )
        df = pd.read_csv("immunogenicity.csv")[["glycan", "immunogenicity"]]
        df.rename(columns={"immunogenicity": "label"}, inplace=True)
        df["split"] = np.random.choice(["train", "val", "test"], df.shape[0], p=[0.7, 0.2, 0.1])
        df.to_csv(p, sep="\t", index=False)
    return p


def get_glycosylation():
    if not (p := Path("glycosylation.tsv")).exists():
        urllib.request.urlretrieve(
            "https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_properties.csv",
            "glycosylation.csv"
        )
        df = pd.read_csv("glycosylation.csv")[["glycan", "link"]]
        df.dropna(inplace=True)
        df["label"] = list(np.array(pd.get_dummies(df["link"]), dtype=int))
        df["split"] = np.random.choice(["train", "val", "test"], df.shape[0], p=[0.7, 0.2, 0.1])
        df.to_csv(p, sep="\t", index=False)
    return p


taxonomy_classes: Dict[str, RawDataInfo] = {
    "Domain": (5, "classification"),
    "Subdomain": (5, "classification"),
    "Kingdom": (1, "classification"),
    "Phylum": (1, "classification"),
    "Class": (1, "multilabel"),
    "Order": (1, "multilabel"),
    "Family": (1, "multilabel"),
    "Genus": (1, "multilabel"),
    "Species": (1, "multilabel"),
    "Immunogenicity": (2, "classification"),
    "Glycosylation": (3, "classification"),
}


def get_dataset(dataset_name) -> DatasetInfo:
    name_fracs = dataset_name.split("_")
    match name_fracs[0]:
        case "Taxonomy":
            path = get_taxonomic_level(name_fracs[1])
            n_classes, task = taxonomy_classes[name_fracs[1]]
        case "Immunogenicity":
            path = get_immunogenicity()
            n_classes, task = taxonomy_classes["Immunogenicity"]
        case "Glycosylation":
            path = get_glycosylation()
            n_classes, task = taxonomy_classes["Glycosylation"]
        case _:
            raise ValueError(f"Unknown dataset {dataset_name}.")
    return {
        "filepath": path,
        "num_classes": n_classes,
        "task": task
    }
