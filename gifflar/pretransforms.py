import copy
from typing import Tuple, Literal, Optional

import torch
from glycowork.glycan_data.loader import lib
from glycowork.motif.graph import glycan_to_nxGraph
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric import transforms as T
from torch_geometric.transforms import Compose
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.utils import from_networkx

from gifflar.utils import bond_map, lib_map, atom_map


class RootTransform(BaseTransform):
    def __init__(self, **kwargs):
        pass


class GIFFLARTransform(RootTransform):
    def __call__(self, data):
        data["atoms"].x = torch.tensor([
            atom_map.get(atom.GetAtomicNum(), len(atom_map)) for atom in data["mol"].GetAtoms()
        ])
        data["atoms"].num_nodes = len(data["atoms"].x)
        data["bonds"].x = []
        data["atoms", "coboundary", "atoms"].edge_index = []
        data["atoms", "to", "bonds"].edge_index = []
        data["bonds", "to", "monosacchs"].edge_index = []
        for bond in data["mol"].GetBonds():
            data["bonds"].x.append(bond_map.get(bond.GetBondType(), 0))
            data["atoms", "coboundary", "atoms"].edge_index += [
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            ]
            data["atoms", "to", "bonds"].edge_index += [
                (bond.GetBeginAtomIdx(), bond.GetIdx()),
                (bond.GetEndAtomIdx(), bond.GetIdx())
            ]
            data["bonds", "to", "monosacchs"].edge_index.append(
                (bond.GetIdx(), bond.GetIntProp("mono_id"))
            )
        data["bonds"].x = torch.tensor(data["bonds"].x)
        data["bonds"].num_nodes = len(data["bonds"].x)
        data["atoms", "coboundary", "atoms"].edge_index = torch.tensor(data["atoms", "coboundary", "atoms"].edge_index,
                                                                       dtype=torch.long).T
        data["atoms", "to", "bonds"].edge_index = torch.tensor(data["atoms", "to", "bonds"].edge_index,
                                                               dtype=torch.long).T
        data["bonds", "to", "monosacchs"].edge_index = torch.tensor(data["bonds", "to", "monosacchs"].edge_index,
                                                                    dtype=torch.long).T
        data["bonds", "boundary", "bonds"].edge_index = torch.tensor(
            [(bond1.GetIdx(), bond2.GetIdx()) for atom in data["mol"].GetAtoms() for bond1 in atom.GetBonds()
             for bond2 in atom.GetBonds() if bond1.GetIdx() != bond2.GetIdx()], dtype=torch.long).T
        data["bonds", "coboundary", "bonds"].edge_index = torch.tensor(
            [(bond1, bond2) for ring in data["mol"].GetRingInfo().BondRings() for bond1 in ring for bond2 in ring if
             bond1 != bond2], dtype=torch.long).T
        data["monosacchs"].x = torch.tensor([
            lib_map.get(data["tree"].nodes[node]["type"].name, len(lib_map)) for node in data["tree"].nodes
        ])
        data["monosacchs"].num_nodes = len(data["monosacchs"].x)
        data["monosacchs", "boundary", "monosacchs"].edge_index = []
        for a, b in data["tree"].edges:
            data["monosacchs", "boundary", "monosacchs"].edge_index += [(a, b), (b, a)]
        data["monosacchs", "boundary", "monosacchs"].edge_index = torch.tensor(
            data["monosacchs", "boundary", "monosacchs"].edge_index, dtype=torch.long).T
        return data


class GNNGLYTransform(RootTransform):
    def __call__(self, data):
        data["gnngly_x"] = torch.tensor([[
            min(atom.GetAtomicNum(), 100),
            min(atom.GetChiralTag(), 3),
            min(atom.GetDegree(), 12),
            min(atom.GetFormalCharge(), 4),
            min(atom.GetTotalNumHs(), 4),
            min(atom.GetHybridization(), 4),
        ] for atom in data["mol"].GetAtoms()])
        data["gnngly_num_nodes"] = len(data["gnngly_x"])
        data["gnngly_edge_index"] = copy.deepcopy(data["atoms", "coboundary", "atoms"].edge_index)
        return data


class ECFPTransform(RootTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ecfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    def __call__(self, data):
        data["fp"] = torch.tensor(self.ecfp.GetFingerprint(data["mol"]), dtype=torch.float).reshape(1, -1)
        return data


class SweetNetTransform(RootTransform):
    def __init__(self, lib, **kwargs):
        super().__init__(**kwargs)
        self.lib = lib

    def __call__(self, data):
        try:
            d = from_networkx(glycan_to_nxGraph(data["IUPAC"], self.lib))
        except:
            d = from_networkx(glycan_to_nxGraph("Gal", lib))
        data["sweetnet_x"] = torch.tensor(d["labels"]).reshape(-1, 1)
        data["sweetnet_num_nodes"] = len(data["sweetnet_x"])
        data["sweetnet_edge_index"] = d.edge_index
        return data


def get_pretransforms(**kwargs) -> [T.Compose]:
    return Compose([
        GIFFLARTransform(**kwargs),
        GNNGLYTransform(**kwargs),
        ECFPTransform(**kwargs),
        SweetNetTransform(lib=lib, **kwargs),
    ])
