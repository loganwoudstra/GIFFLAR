import pickle

from gifflar.data import GlycanStorage

with open("collected.pkl", "rb") as f:
    _, unique_glycans, _ = pickle.load(f)

gs = GlycanStorage("C:/Users/joere/Desktop")
print("Loaded GlycanStorage:", len(gs.data))

data = {}
for i, iupac in enumerate(unique_glycans):
    try:
        print(f"\r{i}", end="")
        res = gs.query(iupac)
        if res:
            data[iupac] = res["smiles"]
    except Exception as e:
        print(e)

print(len(data))
with open("glycan_smiles.pkl", "wb") as f:
    pickle.dump(data, f)
