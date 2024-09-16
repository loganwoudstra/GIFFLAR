import sys

from pathlib import Path

import numpy as np
from glycowork.glycan_data.loader import df_species
from glycowork.motif.graph import glycan_to_nxGraph
import pandas as pd
import torch

from gifflar.data.utils import GlycanStorage

BONDS = {
    "alpha_bond": "C[C@H](OC)CC",
    "beta_bond": "C[C@@H](OC)CC",
    "nostereo_bond": "CC(OC)CC"
}

gs = GlycanStorage("/home/daniel/Data1/roman/GIFFLAR/data_pret")

def parse_mono(filepath, iupac):
    mono = dict()
    bonds = set()

    monos_text = ""
    bonds_text = ""

    g = glycan_to_nxGraph(iupac)

    for n in g.nodes:
        node = g.nodes[n]
        if n % 2 == 0:  # monosaccharide
            r = gs.query(node["string_labels"])
            if r is None:  # return from processing function
                pass
            if "," in node["string_labels"]:
                m = f"\"{node['string_labels']}\""
            else:
                m = node["string_labels"]
            mono[m] = r["smiles"]
            monos_text += f"\n{n // 2 + 1} {m}"
        else:
            if "a" in node["string_labels"]:  # alpha_bond
                bond_type = "alpha_bond"
            elif "b" in node["string_labels"]:
                bond_type = "beta_bond"
            else:
                bond_type = "nostereo_bond"
            bonds.add(bond_type)
            N = list(g.neighbors(n))
            bonds_text += f"\n{min(N) // 2 + 1} {max(N) // 2 + 1} {bond_type}"

    print("SMILES", file=filepath)
    for iupac, smiles in mono.items():
        print(iupac, smiles, file=filepath)
    for bond in bonds:
        print(bond, BONDS[bond], file=filepath)
    print("\nMONOMERS", end="", file=filepath)
    print(monos_text, file=filepath)
    print("\nBONDS", end="", file=filepath)
    print(bonds_text, file=filepath)
    return mono


def parse_level(base: Path, prep_folder: Path, level: str, filepath: Path):
    graphs = base / "graphs"
    graphs.mkdir(exist_ok=True, parents=True)
    valid = {}
    labels = {}
    mode = "T"
    if level == "Immunogenicity":
        mode = "I"
        label_map = {}
        with open(filepath.parent / (filepath.stem + "_classes.tsv"), "r") as f:
            for line in f.readlines():
                v, k = line.strip().split("\t")[:2]
                label_map[int(k)] = "Yes" if float(v) > 0.5 else "No"
        print(label_map)
    if level == "Glycosylation":
        mode = "G"
        label_map = {}
        with open(filepath.parent / (filepath.stem + "_classes.tsv"), "r") as f:
            for line in f.readlines():
                v, k = line.strip().split("\t")[:2]
                label_map[int(k)] = v
        print(label_map)

    for split in {"train", "val", "test"}:
        for data in torch.load(prep_folder / f"{split}.pt")[0]:
            valid[data["IUPAC"]] = split
            if mode == "T":
                labels[data["IUPAC"]] = data["y_oh"]
            else:
                labels[data["IUPAC"]] = label_map[data["y"].item()]

    dataset = pd.read_csv(filepath, sep="\t")
    classes = np.array([x for x in dataset.columns if x not in {"IUPAC", "split"}], dtype=str)

    monos = dict()
    seen = set()
    if mode == "T":
        df_species["ID"] = [f"GID{i + 1:05d}" for i in range(len(df_species))]
        df_species.rename(columns={"glycan": "Glycan"}, inplace=True)
        df_species.drop(columns=[x for x in df_species.columns if x not in {"ID", "Glycan"}], inplace=True)
        df = df_species
    else:
        df = dataset
        df["ID"] = [f"GID{i + 1:05d}" for i in range(len(df))]
        df.rename(columns={"IUPAC": "Glycan"}, inplace=True)
        df.drop(columns=[x for x in df.columns if x not in {"ID", "Glycan"}], inplace=True)
    mask = [False for _ in range(len(df))]
    
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"\rParsing {i}", end="")
        if row["Glycan"] not in valid or row["Glycan"] in seen:
            continue
        seen.add(row["Glycan"])

        mask[i] = True
        with open(graphs / f"{row['ID']}_graph.txt", "w") as f:
            monos.update(parse_mono(f, row["Glycan"]))

    df = df[mask]
    df["split"] = df["Glycan"].map(valid)
    if mode == "T":
        l = torch.cat([labels[x] for x in df["Glycan"].values], dim=0).numpy()
        df[level] = [", ".join(classes[x.astype(bool)]) for x in l]
    else:
        df[level] = df["Glycan"].map(labels)
    df.to_csv(base / "multilabel.txt", index=False)

    with open(base / "bonds.txt", "w") as f:
        print("Molecule,SMILES", file=f)
        for bond, smiles in BONDS.items():
            print(bond, smiles, file=f, sep=",")

    with open(base / "monos.txt", "w") as f:
        print("Molecule,SMILES", file=f)
        for mono, smiles in monos.items():
            print(mono, smiles, file=f, sep=",")


if __name__ == '__main__':
    base, prep_folder, level, filepath = sys.argv[1:5]
    parse_level(Path(base), Path(prep_folder), level, Path(filepath))
