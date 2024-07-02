from pathlib import Path

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
