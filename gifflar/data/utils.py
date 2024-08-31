import copy
import pickle
from pathlib import Path
from typing import Optional
import re

import glyles
import networkx as nx
import rdkit
from glyles.glycans.factory.factory import MonomerFactory
from glyles.glycans.poly.merger import Merger
from rdkit import Chem
from torch_geometric.data import HeteroData
from rdkit.Chem import rdDepictor, BondType


def clean_tree(tree: nx.Graph) -> Optional[nx.Graph]:
    """
    Clean the tree from unnecessary node features and store only the IUPAC name.

    Args:
        tree: The tree to clean

    Returns:
        The cleaned tree
    """
    for node in tree.nodes:
        attributes = copy.deepcopy(tree.nodes[node])
        if "type" in attributes and isinstance(attributes["type"], glyles.glycans.mono.monomer.Monomer):
            tree.nodes[node].clear()
            tree.nodes[node].update({"iupac": "".join([x[0] for x in attributes["type"].recipe]),
                                     "name": attributes["type"].name, "recipe": attributes["type"].recipe})
        else:
            return None
    return tree


def iupac2mol(iupac: str) -> Optional[HeteroData]:
    """
    Convert a glycan stored given as IUPAC-condensed string into an RDKit molecule while keeping the information of
    which atom and which bond belongs to which monosaccharide.

    Args:
        iupac: The IUPAC-condensed string of the glycan to convert

    Returns:
        A HeteroData object containing the IUPAC string, the SMILES representation, the RDKit molecule, and the
        monosaccharide tree
    """
    # convert the IUPAC string using GlyLES
    glycan = glyles.Glycan(iupac)
    # print(glycan.get_smiles())

    # get it's underlying monosaccharide-tree
    tree = glycan.parse_tree

    # re-merge the monosaccharide tree using networkx graphs to keep the assignment of atoms and bonds to monosacchs.
    merged = S3NMerger(MonomerFactory()).merge(tree, glycan.root_orientation, glycan.start)
    mol = nx2mol(merged)

    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    Chem.WedgeMolBonds(mol, mol.GetConformer())

    smiles = Chem.MolToSmiles(mol)
    if len(smiles) < 10 or not isinstance(tree, nx.Graph):
        return None

    tree = clean_tree(tree)

    data = HeteroData()
    data["IUPAC"] = iupac
    data["smiles"] = smiles
    data["mol"] = mol
    data["tree"] = tree
    return data


def nx2mol(G: nx.Graph, sanitize=True) -> Chem.Mol:
    """
    Convert a molecules from a networkx.Graph to RDKit.

    Args:
        G: The graph representing a molecule
        sanitize: A bool flag indicating to sanitize the resulting molecule (should be True for "production mode" and
            False when debugging this function)

    Returns:
        The converted, sanitized molecules in RDKit represented by the input graph
    """
    # Create the molecule
    mol = Chem.RWMol()

    # Extract the node attributes
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    mono_ids = nx.get_node_attributes(G, 'mono_id')

    # Create all atoms based on their representing nodes
    node_to_idx = {}
    for node in G.nodes():
        a = Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetIntProp("mono_id", mono_ids[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    # Extract the edge attributes
    bond_types = nx.get_edge_attributes(G, 'bond_type')
    mono_bond_ids = nx.get_edge_attributes(G, 'mono_id')

    # Connect the atoms based on the edges from the graph
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mono_idx = mono_bond_ids[first, second]
        idx = mol.AddBond(ifirst, isecond, bond_type) - 1
        mol.GetBondWithIdx(idx).SetIntProp("mono_id", mono_idx)

    # print(Chem.MolToSmiles(mol))
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol


def graph_removal(G: nx.Graph, del_node_idx: int) -> nx.Graph:
    """
    Remove a node from a graph and reindex the nodes and edges accordingly.

    Args:
        G: The graph to remove the node from
        del_node_idx: The index of the node to remove

    Returns:
        The graph with the node removed and the indices reindexed
    """
    H = nx.Graph()
    for node in G.nodes:
        if node < del_node_idx:
            H.add_node(node, **G.nodes[node])
        elif node > del_node_idx:
            H.add_node(node - 1, **G.nodes[node])
    for edge in G.edges:
        if del_node_idx in edge:
            continue
        start, end = edge
        if start > del_node_idx:
            start -= 1
        if end > del_node_idx:
            end -= 1
        H.add_edge(start, end, **G.edges[edge])
    return H


def connect_nx(atom: int, child_start: int, kids: nx.Graph, me: nx.Graph, original_atom: int) -> nx.Graph:
    """
    Connect a monosaccharide to a glycan.

    Args:
        atom: Name of the placeholder atom in me
        child_start: Idx of the atom to connect to (the one to be replaced theoretically) in kids
        kids: nx object of the children downstream the current connection
        me: nx object of the monosaccharide to prepend
        original_atom: The type of the original atom to be replaced in the monosaccharide

    Returns:
        The merged nx object
    """
    me_idx = [k for k, v in nx.get_node_attributes(me, "atomic_num").items() if v == atom][0]
    anchor_c = list(kids.neighbors(child_start))[0]

    # anchor_p = list(me.neighbors(me_idx))[0]
    # me.nodes[anchor_p]["atomic_num"] = 32       # Ge
    # me.nodes[me_idx]["atomic_num"] = 34         # Se
    # kids.nodes[anchor_c]["atomic_num"] = 50     # Sn
    # kids.nodes[child_start]["atomic_num"] = 52  # Te

    G = nx.disjoint_union(me, kids)

    # G.add_edge(me_idx, len(me) + child_start, bond_type=BondType.SINGLE, mono_id=nx.get_node_attributes(kids, "mono_id")[child_start])

    G.add_edge(me_idx, len(me) + anchor_c, bond_type=BondType.SINGLE,
               mono_id=nx.get_node_attributes(kids, "mono_id")[child_start])
    G.remove_node(len(me) + child_start)

    G.nodes[me_idx]["atomic_num"] = original_atom
    # print(me_idx)
    return graph_removal(G, len(me) + child_start)


class S3NMerger(Merger):
    def merge(self, t: nx.Graph, root_orientation: str = "n", start: int = 100) -> nx.Graph:
        """
        Merge a tree into a single molecule.

        Args:
            t: Tree representing the glycan to compute the whole SMILES representation for
            root_orientation: The orientation of the root monomer
            start: The starting index for the molecule

        Returns:
            The merged molecule
        """
        # first mark the atoms that will be replaced in a binding of two monomers
        self.__mark(t, 0, f"({root_orientation}1-?)")
        # then merge the tree into a single molecule
        return self.__merge(t, 0, 0)

    def __mark(self, t: nx.Graph, node: int, p_edge: Optional[str] = None) -> None:
        """
        Recursively mark in every node of the molecule which atoms are being replaced by bound monomers.

        Args:
            t: Tree representing the glycan to compute the whole SMILES representation for.
            node: ID of the node to work on in this method
            p_edge: edge annotation to parent monomer
        """
        # get children nodes
        children = [x[1] for x in t.edges(node)]

        # set chirality of atom binding parent
        if p_edge is not None and t.nodes[node]["type"].is_non_chiral():
            t.nodes[node]["type"] = t.nodes[node]["type"].to_chirality(p_edge[1], self.factory)

        # check for validity of the tree, ie if it's a leaf (return, nothing to do) or has too many children (Error)
        if len(children) == 0:  # leaf
            return
        if len(children) > 4:  # too many children
            raise NotImplementedError("Glycans with maximal branching factor greater then 3 not implemented.")

        # iterate over the children and the atoms used to mark binding atoms in my structure
        for child, atom in zip(children, t.nodes[node]["type"].get_dummy_atoms()):
            binding = re.findall(r'\d+', t.get_edge_data(node, child)["type"])[1]

            t.nodes[node]["type"].mark(int(binding), *atom)
            self.__mark(t, child, t.get_edge_data(node, child)["type"])

    def __merge(self, t: nx.Graph, node: int, start: int, ring_index: Optional[int] = None) -> nx.Graph:
        """
        Recursively merge a tree into a single molecule.

        Args:
            t: Tree representing the glycan to compute the whole SMILES representation for
            node: ID of the node to work on in this method
            start: The starting index for the molecule
            ring_index: The index of the ring to close

        Returns:
            The merged molecule
        """
        children = [x[1] for x in t.edges(node)]
        me = mol2nx(t.nodes[node]["type"].structure, node)

        if len(children) == 0:
            return me

        for child, (o_atom, n_atom) in zip(children, t.nodes[node]["type"].get_dummy_atoms()):
            binding = re.findall(r'\d+', t.get_edge_data(node, child)["type"])[0]
            child_start = t.nodes[child]["type"].root_atom_id(int(binding))

            # get the SMILES of this child and plug it in the current own SMILES
            child = self.__merge(t, child, child_start)
            if o_atom[0] in nx.get_node_attributes(me, "atomic_num").values():
                me = connect_nx(o_atom[0], child_start, child, me, 8)
            elif n_atom[0] in nx.get_node_attributes(me, "atomic_num").values():
                me = connect_nx(n_atom[0], child_start, child, me, 7)
        return me


def mol2nx(mol: rdkit.Chem.Mol, node: int) -> nx.Graph:
    """
    Convert a monosaccharide to a networkx graph.

    Args:
        mol: The molecule to be converted
        node: The monosaccharide ID in the whole glycan

    Returns:
        A networkx.Graph representing the molecule with the same properties
    """
    G = nx.Graph()

    # Convert all the atoms, explicit Hs and Hybridization have to be dropped because they interfere with further steps)
    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
            chiral_tag=atom.GetChiralTag(),
            is_aromatic=atom.GetIsAromatic(),
            mono_id=node,
        )
    # Convert all the bonds
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=bond.GetBondType(),
            mono_id=node,
        )
    return G


class GlycanStorage:
    def __init__(self, path: Path | str | None = None):
        """
        Initialize the wrapper around a dict.

        Args:
            path: Path to the directory. If there's a glycan_storage.pkl, it will be used to fill this object,
                otherwise, such file will be created.
        """
        self.path = Path(path or "data") / "glycan_storage.pkl"

        # Fill the storage from the file
        self.data = self._load()

    def close(self) -> None:
        """
        Close the storage by storing the dictionary at the location provided at initialization.
        """
        with open(self.path, "wb") as out:
            pickle.dump(self.data, out)

    def query(self, iupac: str) -> Optional[HeteroData]:
        """
        Query the storage for a IUPAC string.

        Args:
            iupac: The IUPAC string of the query glycan

        Returns:
            A HeteroData object corresponding to the IUPAC string or None, if the IUPAC string could not be processed.
        """
        if iupac not in self.data:
            try:
                self.data[iupac] = iupac2mol(iupac)
            except:
                self.data[iupac] = None
        return copy.deepcopy(self.data[iupac])

    def _load(self) -> dict[str, Optional[HeteroData]]:
        """
        Load the internal dictionary from the file, if it exists, otherwise, return an empty dict.

        Returns:
            The loaded (if possible) or an empty dictionary
        """
        if self.path.exists():
            with open(self.path, "rb") as f:
                return pickle.load(f)
        return {}
