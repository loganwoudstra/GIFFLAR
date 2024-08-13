import urllib.request
from pathlib import Path
from typing import Dict, Optional, Literal

import numpy as np
import pandas as pd
from glycowork.glycan_data.loader import df_species as taxonomy
from glyles import convert
from tqdm import tqdm
# from datasail.sail import datasail


def iupac2smiles(iupac: str) -> Optional[str]:
    """
    Convert IUPAC-condensed representations to SMILES strings (or None if the smiles string cannot be valid.
    """
    if any([x in iupac for x in ["?", "{", "}"]]):
        return None
    try:
        smiles = convert(iupac)[0][1]
        if len(smiles) < 10:
            return None
        return smiles
    except:
        return None


def get_taxonomy() -> pd.DataFrame:
    """
    Download full taxonomy data, process it, and save it as a tsv file.
    """
    if not (p := Path("taxonomy.tsv")).exists():
        mask = []
        # convert to IUPAC to SMILES and build a mask to remove not-convertable molecules.
        for i in tqdm(taxonomy["glycan"]):
            smiles = iupac2smiles(i)
            mask.append(smiles is not None)
        tax = taxonomy[mask]
        tax.to_csv(p, sep="\t", index=False)
    return pd.read_csv(p, sep="\t")


def get_taxonomic_level(
        level: Literal["Domain", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
) -> Path:
    """
    Extract taxonomy data at a specific level, process it, and save it as a tsv file.

    Returns:
        Path to the TSV file storing the processed dataset.
    """
    if not (p := Path(f"taxonomy_{level}.tsv")).exists():
        # Chop to taxonomic level of interest and remove invalid rows
        tax = get_taxonomy()[["glycan", level]]
        tax.rename(columns={"glycan": "IUPAC"}, inplace=True)
        tax[tax[level] == "undetermined"] = np.nan
        tax.dropna(inplace=True)

        # One-hot encode the individual classes and collate them for glycans that are the same
        tax = pd.concat([tax["IUPAC"], pd.get_dummies(tax[level])], axis=1)
        tax = tax.groupby('IUPAC').agg("sum").reset_index()

        # Chop prediction values to 0 and 1
        classes = [x for x in tax.columns if x != "IUPAC"]
        tax[classes] = tax[classes].applymap(lambda x: min(1, x))

        # Remove all columns with less than 5 members. 3 would also work, but that has potentially not valid split
        # tax.drop(np.array(classes)[tax[classes].sum() < 5], axis="columns", inplace=True)
        # classes = [x for x in tax.columns if x != "IUPAC"]
        # remove all glycans that are not part of any class in the columns
        # tax = tax[tax[classes].sum(axis=1) > 0]
        # print(tax.shape)

        # Apply a random split
        # strats = dict(zip(tax["IUPAC"], tax[classes].apply(lambda row: "".join([str(row[c]) for c in classes]), axis=1)))
        # e_split, _, _ = datasail(
        #     techniques=["I1e"],
        #     splits=[7, 2, 1],
        #     names=["train", "val", "test"],
        #     e_type="O",
        #     e_data=dict(zip(tax["IUPAC"], tax["IUPAC"])),
        #     e_strat=strats,
        #     delta=0.3,
        # )
        # tax["split"] = tax["IUPAC"].map(e_split["I1e"][0])
        print(tax.shape)
        tax["split"] = np.random.choice(["train", "val", "test"], tax.shape[0], p=[0.7, 0.2, 0.1])
        mask = ((tax[tax["split"] == "train"][classes].sum() == 0) |
                (tax[tax["split"] == "val"][classes].sum() == 0) |
                (tax[tax["split"] == "test"][classes].sum() == 0))
        tax.drop(columns=np.array(classes)[mask], inplace=True)
        classes = [x for x in tax.columns if x not in {"IUPAC", "split"}]
        tax = tax[tax[classes].sum(axis=1) > 0]
        print(tax.shape)
        # tax.drop(level, axis=1, inplace=True)
        tax.to_csv(p, sep="\t", index=False)
    return p


def get_immunogenicity() -> Path:
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
        # Download the data
        urllib.request.urlretrieve(
            "https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_immunogenicity.csv",
            "immunogenicity.csv"
        )

        # Process the data and remove unnecessary columns
        df = pd.read_csv("immunogenicity.csv")[["glycan", "immunogenicity"]]
        df.rename(columns={"glycan": "IUPAC"}, inplace=True)
        df.dropna(inplace=True)

        # One-hot encode the individual classes and collate them for glycans that are the same
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
