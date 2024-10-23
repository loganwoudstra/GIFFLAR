import pickle
import random

from tqdm import tqdm
import numpy as np
from glycowork.glycan_data.loader import glycan_binding as lgi


# Use stack to convert to a Series with a MultiIndex
lgi.index = lgi["target"]
lgi.drop(columns=["target", "protein"], inplace=True)
s = lgi.stack()

glycans = {f"Gly{i:04d}": iupac for i, iupac in enumerate(lgi.columns)}
glycans.update({iupac: f"Gly{i:04d}" for i, iupac in enumerate(lgi.columns)})

lectins = {f"Lec{i:04d}": aa_seq for i, aa_seq in enumerate(lgi.index)}
lectins.update({aa_seq: f"Lec{i:04d}" for i, aa_seq in enumerate(lgi.index)})

# Convert the Series to a list of triplets (row, col, val)
data = []
splits = np.random.choice(["train", "val", "test"], len(s), p=[0.7, 0.2, 0.1])
for i, ((aa_seq, iupac), val) in tqdm(enumerate(s.items())):
    data.append((lectins[aa_seq], glycans[iupac], val, splits[i]))

data = random.sample(data, int(len(data) * 0.20))

with open("lgi_data_20.pkl", "wb") as f:
    pickle.dump((data, lectins, glycans), f)

