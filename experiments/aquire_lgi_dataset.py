from glycowork.glycan_data.loader import glycan_binding as lgi

import pandas as pd
import numpy as np


# Use stack to convert to a Series with a MultiIndex
s = lgi.stack()

# Convert the Series to a list of triplets (row, col, val)
triplets = [(i, j, val) for (i, j), val in s.items()]

print(len(triplets))
