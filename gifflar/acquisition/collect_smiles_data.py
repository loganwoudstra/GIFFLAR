import pickle
from pathlib import Path

from glyles import convert

from gifflar.data.utils import GlycanStorage

glycans_path = Path("subglycans.pkl")

if not glycans_path.exists():
    import gifflar.acquisition.collect_pretrain_data

print("Extracting SMILES data...\n=========================")

with open(glycans_path, "rb") as f:
    iupacs = pickle.load(f)

root = Path("/") / "home" / "lwoudstr" / "scratch" / "GIFFLAR" / "data_pret"

gs = GlycanStorage(root)
data = []
for iupac in iupacs:
    try:
        if gs.query(iupac) is not None:
            data.append(iupac)
    except Exception as e:
        pass

gs.close()

with open(root / "expanded_glycans.txt", "w") as f:
    print(*data, sep="\n", file=f)
