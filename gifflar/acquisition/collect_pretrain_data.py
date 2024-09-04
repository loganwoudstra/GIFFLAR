import copy
import hashlib
import pickle
from pathlib import Path

from glycowork.motif.graph import glycan_to_nxGraph, graph_to_string

glycans_path = Path("glycans.pkl")
if not glycans_path.exists():
    import collect_glycan_data


def node_label_hash(label):
    """Hash function for individual node labels."""
    return int(hashlib.sha256(str(label).encode()).hexdigest(), 16)


def wl_relabel(G, num_iterations=3):
    """Weisfeiler-Lehman relabeling for graph G with node aggregation."""
    # Initialize labels with node features if available, or with node degrees
    labels = {node: node_label_hash(G.nodes[node].get('string_labels', G.degree(node)))
              for node in G.nodes}

    for _ in range(num_iterations):
        new_labels = {}
        for node in G.nodes:
            # Aggregate hashes of the neighbors
            neighbor_hashes = [labels[neighbor] for neighbor in G.neighbors(node)]
            aggregated_hash = sum(neighbor_hashes)  # Sum ensures permutation invariance
            new_labels[node] = node_label_hash(labels[node] + aggregated_hash)
        labels = new_labels

    return labels


def graph_hash(G, num_iterations=3):
    """Compute a permutation-invariant hash of the entire graph."""
    labels = wl_relabel(G, num_iterations)
    # Aggregate node labels to create the final graph hash
    final_hash = sum(labels.values())  # Summing ensures permutation invariance
    return hashlib.sha256(str(final_hash).encode()).hexdigest()


def cut_and_add(glycan):
    h = graph_hash(glycan)
    if h in known:
        return
    known.add(h)
    
    try:
        known_iupacs.append(graph_to_string(glycan))
    except:
        return

    # check if glycan is a single node
    if len(glycan.nodes()) == 1:
        return

    leafs = [x for x in glycan.nodes() if glycan.degree(x) == 1 and x != 0]
    for x in leafs:
        G = copy.deepcopy(glycan)
        parent = list(G.neighbors(x))[0]
        G.remove_node(x)
        G.remove_node(parent)
        cut_and_add(G)


print("Collecting subglycan data...\n============================")

with open(glycans_path, "rb") as f:
    _, iupacs, _ = pickle.load(f)
iupacs = sorted(iupacs, key=lambda x: x.count("("))

known_iupacs = []
known = set()
for i, iupac in enumerate(iupacs):
    print(f"\r{i}/{len(iupacs)}\t{iupac}", end="")
    if iupac.count("(") == 15:
        print()
        print(f"Stopped calculation due to high complexity after {i} of {len(iupacs)} glycans.")
        break
    try:
        cut_and_add(glycan_to_nxGraph(iupac))
    except:
        pass

with open("subglycans.pkl", "wb") as f:
    pickle.dump(known_iupacs, f)
