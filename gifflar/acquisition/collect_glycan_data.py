from collections import Counter
import urllib.request
import pickle
from pathlib import Path

import numpy as np
from glycowork.glycan_data import loader
from glycowork.motif.graph import graph_to_string, get_possible_topologies, glycan_to_nxGraph
import networkx as nx
from networkx.algorithms.isomorphism import categorical_node_match


print("Collecting glycan data...\n=========================")

# Collect data from all sources in glycowork (assumed to be installed) and CandyCrunch
if not (p := Path("glycans.pkl")).exists():
    urllib.request.urlretrieve('https://github.com/BojarLab/CandyCrunch/raw/main/CandyCrunch/glycans.pkl', 'glycans.pkl')
with open('glycans.pkl', 'rb') as f:
    candy = set(pickle.load(f))
print("Glycans in candycrunch:", len(candy))

sugarbase = set(getattr(loader, 'df_glycan')["glycan"])
print("Glycans in the sugarbase:", len(sugarbase))

# Validate that each source adds to the total number of glycans
iupacs = list(sugarbase.union(candy))
print("Glycans in combined list:", len(iupacs))

expanded = []
for glycan in list(filter(lambda x: "{" in x, iupacs)):
    try:
        expanded += [graph_to_string(g) for g in get_possible_topologies(glycan)]
    except:
        pass
iupacs = list(filter(lambda x: "{" not in x, iupacs)) + expanded
print("Glycans in expanded list:", len(iupacs))

# Convert glycans to networkx graphs and hash them
graphs = [glycan_to_nxGraph(iupac) for iupac in iupacs]

fps = {}
ids = []
for graph in graphs:
    h = hash(tuple(sorted(Counter(nx.get_node_attributes(graph, "string_labels").values()).items())))
    if h not in fps:
        fps[h] = len(fps)
    ids.append(fps[h])
ids = np.array(ids)

# Iterate over all buckets of glycans with the same hash and find isomorphisms
iupacs = np.array(iupacs)
graphs = np.array(graphs, dtype=object)
glycans = []
isomorphisms = []
for i in range(ids.max() + 1):
    print(f"\r{i}", end="")
    mask = ids == i
    if sum(mask) == 1:
        glycans += iupacs[mask].tolist()
        continue
    graph_list = graphs[mask]
    iupac_list = iupacs[mask]
    tree_list = [nx.dfs_tree(g, max(list(g.nodes.keys()))) for g in graph_list]
    is_isomorphic = [False] * len(tree_list)
    for i in range(len(tree_list)):
        if is_isomorphic[i]:
            continue
        glycans.append(iupacs[i])
        for j in range(i + 1, len(tree_list)):
            if nx.is_isomorphic(tree_list[i], tree_list[j]) and nx.is_isomorphic(graph_list[i], graph_list[j], categorical_node_match("string_labels", "")):
                isomorphisms.append((iupacs[i], iupacs[j]))
                is_isomorphic[j] = True
                break

print("\nNumber of non-isomorphic glycans:", len(glycans))

# Visualize a pair of isomorphics glycans
print("\n".join(isomorphisms[0]))

# Store everything in a pickle file, first the unique iupacs, then the glycans, and finally the isomorphic pairs. The
# isomorphic pairs are stored as tuples of iupacs (first the kept one, second the isomorphic, removed one).
with open("glycans.pkl", "wb") as f:
    pickle.dump((iupacs, glycans, isomorphisms), f)
