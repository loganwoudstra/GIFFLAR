import sys
import urllib
import requests
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.wait import WebDriverWait


threetoone = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}


class Residue:
    """Residue class"""

    def __init__(self, line) -> None:
        self.name = line[17:20].strip()
        self.num = int(line[22:26].strip())
        self.chainID = line[21].strip()


class Structure:
    def __init__(self, filename) -> None:
        self.residues = []
        self.parse_file(filename)
        self.chains = {}
        for res in self.residues:
            if res.chainID not in self.chains:
                self.chains[res.chainID] = ""
            self.chains[res.chainID] += threetoone[res.name]

    def parse_file(self, filename) -> None:
        for line in open(filename, "r"):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                self.residues.append(Residue(line))
        self.residues = sorted(self.residues, key=lambda x: x.num)


def pdb2seq(root: Path, pdb_id, chain):
    try:
        (p := root / "pdbs").mkdir(exist_ok=True)
        if not (p / f"{pdb_id}.pdb").exists():
            urllib.request.urlretrieve(
                f"https://files.rcsb.org/download/{pdb_id}.pdb",
                p / f"{pdb_id}.pdb",
            )
        struct = Structure(p / f"{pdb_id}.pdb")
        if (chain is None or np.isnan(chain) or chain == "nan") and len(struct.chains) == 1:
            chain = list(struct.chains.keys())[0]
        if chain not in struct.chains:
            return None
        return struct.chains[chain]
    except:
        return None


def fetch_list_online():
    # Fetch the list of Unilectin sequences from the UniProt database
    doc = BeautifulSoup(requests.get("https://unilectin.unige.ch/curated/structure").content, "html.parser")
    content = doc.find_all("div", id="content-main")[0]
    return [x.text for x in content.find_all("a")[1:-12]]


def fetch_single_online(pdb_id):
    doc = BeautifulSoup(requests.get(f"https://unilectin.unige.ch/unilectin3D/display_structure?pdb={pdb_id}").content, "html.parser")
    entries = doc.find_all("div", class_="input-group")
    return {
        "pdb": pdb_id,
        "resolution": list(entries[3].children)[2].get("value"),
        "uniprot": list(entries[7].children)[3].text.split(" ")[0],
        "chain": None,
        "fold": list(entries[0].children)[2].get("value"),
        "class": list(entries[1].children)[2].get("value"),
        "family": list(entries[2].children)[2].get("value"),
        "origin": list(entries[4].children)[2].get("value"),
        "species": list(entries[5].children)[2].get("value"),
        "ligand": list(entries[10].children)[2].get("value"),
        "monosac": list(entries[11].children)[2].get("value"),
        "iupac": list(entries[12].children)[2].get("value"),
        "source": "UniLectin_web"
    }


def fetch_cbm(root: Path):
    with open(Path(__file__).parent.parent / "datasets" / "CBMcarb.html", "r", encoding="utf-8") as f:
        html = f.read()
    doc = BeautifulSoup(html, "html.parser")
    fields = doc.find_all("div", id="viewer_div")[0].find_all("div", class_="form-group-sm row")
    data = {"pdb": [], "resolution": [], "uniprot": [], "chain": [], "fold": [], "class": [], "family": [],
            "origin": [], "species": [], "ligand": [], "monosac": [], "iupac": [], "source": []}
    for i in range(0, len(fields), 18):
        data["pdb"].append(list(fields[i + 0].children)[1].get("value"))
        data["resolution"].append(list(fields[i + 7].children)[1].get("value"))
        data["uniprot"].append(None)
        data["chain"].append(None)
        data["fold"].append(None)
        data["class"].append(list(fields[i + 4].children)[1].get("value"))
        data["family"].append(list(fields[i + 2].children)[1].get("value"))
        data["origin"].append(list(fields[i + 6].children)[1].get("value"))
        data["species"].append(list(fields[i + 5].children)[1].get("value"))
        data["ligand"].append(list(fields[i + 11].children)[1].get("value"))
        data["monosac"].append(None)
        data["iupac"].append(list(fields[i + 9].children)[1].get("value"))
        data["source"].append("CBMcarb")
    print(f"Loaded {len(data['pdb'])} interactions from CBMcarb")
    return data


def fetch_single_csv(pdb_id, df):
    output = df[df["pdb"] == pdb_id][["pdb", "resolution", "uniprot", "chain", "fold", "class", "family", "origin", "species", "ligand", "monosac", "iupac"]].to_dict()
    output["source"] = "UniLectin_csv"
    return output


def append_dict(d1, d2):
    for key in d1.keys():
        d1[key].append(d2[key])
    return d1


def collect_unilectin_cbm(tmp_results: Path):
    df = pd.read_csv(Path(__file__).parent.parent / "datasets" / "unilectin3D.csv")
    print(f"Loaded {len(df)} interactions from offline-UniLectin3D")
    pdb_ids = df["pdb"].to_list()
    online_pdb_ids = set(fetch_list_online())
    print(f"Found {len(online_pdb_ids)} interactions online")
    print(f"Found {len(all_ids := set(pdb_ids).union(online_pdb_ids))} unique interactions in both datasets")
    data = {"pdb": [], "resolution": [], "uniprot": [], "chain": [], "fold": [], "class": [], "family": [], "origin": [], "species": [], "ligand": [], "monosac": [], "iupac": []}  # fetch_cbm(root)
    print("Fetching data...")
    for pdb_id in tqdm(all_ids):
        try:
            if pdb_id in data["pdb"]:
                continue
            data = append_dict(
                data,
                fetch_single_online(pdb_id) if pdb_id in online_pdb_ids else fetch_single_csv(pdb_id, df)
            )
        except Exception as e:
            print(f"Error when fetching {pdb_id}")
            print(e)
    print(f"Collected {len(data['pdb'])} interactions in total")
    pd.DataFrame(data).to_csv(tmp_results, sep="\t", index=False)


def clear_iupac(iupac):
    iupac = iupac.replace(" ", "")
    iupac = iupac.replace("LG", "L-G")
    return iupac


def build_unilectin(tmp_results: Path):
    root = tmp_results.parent
    db = pd.read_csv(tmp_results, sep="\t")
    db.rename(columns={"pdb": "PDB_ID", "uniprot": "UNIPROT_ID", "iupac": "IUPAC"}, inplace=True)
    db.dropna(subset=["IUPAC", "PDB_ID"], inplace=True)

    data = {"IUPAC": [], "seq": [], "PDB_ID": [], "UNIPROT_ID": []}
    for i, (_, row) in enumerate(db.iterrows()):
        print(f"\r{i}/{len(db)}", end="")
        if "; " in row["IUPAC"]:
            iupacs = row["IUPAC"].split("; ")
        else:
            iupacs = [row["IUPAC"]]
        for iupac in iupacs:
            if (seq := pdb2seq(root, row["PDB_ID"], row["chain"])) is not None:
                data["IUPAC"].append(iupac)
                data["seq"].append(seq)
                data["PDB_ID"].append(row["PDB_ID"])
                data["UNIPROT_ID"].append(row["UNIPROT_ID"])
    df = pd.DataFrame(data)[["IUPAC", "seq", "PDB_ID", "UNIPROT_ID"]]
    df["split"] = np.random.choice(["val", "test"], len(df), p=[2/3, 1/3])
    df["y"] = 1
    df.to_csv(tmp_results.parent / "unilectin.tsv", sep="\t", index=False)


if __name__ == "__main__":
    tmp_results = Path(sys.argv[1]) / "unilec_cbm_results.tsv"
    tmp_results.parent.mkdir(exist_ok=True, parents=True)
    if not tmp_results.exists():
        collect_unilectin_cbm(tmp_results)
    build_unilectin(tmp_results)
