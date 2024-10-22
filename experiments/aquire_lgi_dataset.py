import pickle

import numpy as np
from glycowork.glycan_data.loader import glycan_binding as lgi


# Use stack to convert to a Series with a MultiIndex
lgi.index = lgi["target"]
lgi.drop(columns=["target", "protein"], inplace=True)
s = lgi.stack()

glycans = {f"Gly{i:04d}": iupac for i, iupac in enumerate(lgi.columns[:-2])}
glycans.update({iupac: f"Gly{i:04d}" for i, iupac in enumerate(lgi.columns[:-2])})

lectins = {f"Lec{i:04d}": aa_seq for i, aa_seq in enumerate(lgi.index)}
lectins.update({aa_seq: f"Lec{i:04d}" for i, aa_seq in enumerate(lgi.index)})

# Convert the Series to a list of triplets (row, col, val)
data = []
splits = np.random.choice(s.index, len(s))
for i, ((aa_seq, iupac), val) in enumerate(s.items()):
    data.append((lectins[aa_seq], glycans[iupac], val, splits[i]))
    if i == 1000:
        break

with open("lgi_data.pkl", "wb") as f:
    pickle.dump((data, lectins, glycans), f)
