import pickle
from pathlib import Path

from glyles import convert

glycans_path = Path("subglycans.pkl")
if not glycans_path.exists():
    import collect_pretrain_data

print("Extracting SMILES data...\n=========================")

with open(glycans_path, "rb") as f:
    iupacs = pickle.load(f)

data = []
for iupac in iupacs:
    try:
        smiles = convert(iupac)
        if smiles is not None and len(smiles) > 10:
            data.append(iupac)
    except:
        pass

with open("pretrain_glycans.txt", "w") as f:
    print(data, sep="\n", file=f)
