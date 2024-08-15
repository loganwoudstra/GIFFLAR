import copy
from typing import Any, Union

import torch
from glycowork.glycan_data.loader import lib
from glycowork.motif.graph import glycan_to_nxGraph
try:
    from rdkit.Chem import rdFingerprintGenerator
    RDKIT_GEN = True
except ImportError:
    from rdkit.Chem import AllChem
    RDKIT_GEN = False
from torch_geometric import transforms as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import Compose, AddLaplacianEigenvectorPE
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.utils import from_networkx, to_dense_adj
from tqdm import tqdm

from gifflar.utils import bond_map, lib_map, atom_map


def split_hetero_graph(data):
    atoms_data = Data(
        edge_index=data["atoms", "coboundary", "atoms"]["edge_index"],
        num_nodes=data["atoms"]["num_nodes"],
    )
    bond_data = Data(
        edge_index=data["bonds", "boundary", "bonds"]["edge_index"],
        num_nodes=data["bonds"]["num_nodes"],
    )
    monosacchs_data = Data(
        edge_index=data["monosacchs", "boundary", "monosacchs"]["edge_index"],
        num_nodes=data["monosacchs"]["num_nodes"],
    )
    return atoms_data, bond_data, monosacchs_data


def hetero_to_homo(data):
    bond_edge_index = data["bonds", "boundary", "bonds"]["edge_index"] + data["atoms"]["num_nodes"]
    monosacchs_edge_index = data["bonds", "boundary", "bonds"]["edge_index"] + data["atoms"]["num_nodes"] + \
                            data["bonds"]["num_nodes"]
    return Data(
        edge_index=torch.cat([
            data["atoms", "coboundary", "atoms"]["edge_index"],
            bond_edge_index, monosacchs_edge_index
        ], dim=1),
        num_nodes=data["atoms"]["num_nodes"] + data["bonds"]["num_nodes"] + data["monosacchs"]["num_nodes"]
    )


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
        data["monosacchs"].x = torch.tensor([  # This does not make sense. The monomer-ids are categorical features
            lib_map.get(data["tree"].nodes[node]["name"], len(lib_map)) for node in data["tree"].nodes
        ])
        data["monosacchs"].num_nodes = len(data["monosacchs"].x)
        data["monosacchs", "boundary", "monosacchs"].edge_index = []
        for a, b in data["tree"].edges:
            data["monosacchs", "boundary", "monosacchs"].edge_index += [(a, b), (b, a)]
        data["monosacchs", "boundary", "monosacchs"].edge_index = torch.tensor(
            data["monosacchs", "boundary", "monosacchs"].edge_index, dtype=torch.long).T
        return data


class GNNGLYTransform(RootTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.atom_encoder = torch.eye(101)
        self.chiral_encoder = torch.eye(4)
        self.degree_encoder = torch.eye(13)
        self.charge_encoder = torch.eye(5)
        self.h_encoder = torch.eye(5)
        self.hybrid_encoder = torch.eye(5)

    def __call__(self, data):
        data["gnngly_x"] = torch.stack([torch.concat([
            self.atom_encoder[min(atom.GetAtomicNum(), 100)],
            self.chiral_encoder[min(atom.GetChiralTag(), 3)],
            self.degree_encoder[min(atom.GetDegree(), 12)],
            self.charge_encoder[min(atom.GetFormalCharge(), 4)],
            self.h_encoder[min(atom.GetTotalNumHs(), 4)],
            self.hybrid_encoder[min(atom.GetHybridization(), 4)],
        ]) for atom in data["mol"].GetAtoms()])
        data["gnngly_num_nodes"] = len(data["gnngly_x"])
        data["gnngly_edge_index"] = copy.deepcopy(data["atoms", "coboundary", "atoms"].edge_index)
        return data


class ECFPTransform(RootTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if RDKIT_GEN:
            self.ecfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    def __call__(self, data):
        if RDKIT_GEN:
            data["fp"] = torch.tensor(self.ecfp.GetFingerprint(data["mol"]), dtype=torch.float).reshape(1, -1)
        else:
            data["fp"] = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(data["mol"], 2, nBits=1024), dtype=torch.float).reshape(1, -1)
        return data


class SweetNetTransform(RootTransform):
    def __init__(self, glycan_lib=lib, **kwargs: Any):
        """
        Transformation to convert a glycan IUPAC string to a PyG Data object for the SweetNet model.

        Args:
            glycan_lib: The glycan library to use.
            kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.glycan_lib = glycan_lib

    def __call__(self, data):
        try:
            d = from_networkx(glycan_to_nxGraph(data["IUPAC"], self.glycan_lib))
        except:
            d = from_networkx(glycan_to_nxGraph("Gal", self.glycan_lib))
        data["sweetnet_x"] = d["labels"].clone().detach().reshape(-1, 1)
        data["sweetnet_num_nodes"] = len(data["sweetnet_x"])
        data["sweetnet_edge_index"] = d.edge_index
        return data


class LaplacianPE(AddLaplacianEigenvectorPE):
    attr_name = "lap_pe"

    def __init__(self, dim: int, individual: bool = True, **kwargs: Any):
        """
        Args:
            dim: Number of eigenvectors to compute. (k)
            individual: Whether to compute eigenvectors for each node type individually.
            kwargs: Additional arguments.
        """
        super().__init__(k=dim, attr_name=self.attr_name, **kwargs)
        self.max_dim = dim
        self.individual = individual

    def __call__(self, data):
        if self.individual:
            for d, name in zip(split_hetero_graph(data), ["atoms", "bonds", "monosacchs"]):
                if d["num_nodes"] <= self.max_dim:
                    self.k = min(d["num_nodes"] - 1, self.max_dim)
                    if self.k == 0:
                        d[self.attr_name] = torch.tensor([[]])
                    else:
                        super(LaplacianPE, self).forward(d)
                    pad = torch.zeros(d["num_nodes"], self.max_dim - d[self.attr_name].size(1))
                    d[self.attr_name] = torch.cat([d[self.attr_name], pad], dim=1)
                    self.k = self.max_dim
                else:
                    super(LaplacianPE, self).forward(d)
                data[f"{name}_{self.attr_name}"] = d[self.attr_name]
        else:
            d = hetero_to_homo(data)
            super(LaplacianPE, self).forward(d)
            data[f"atoms_{self.attr_name}"] = d[self.attr_name][:data["atoms"]["num_nodes"]]
            data[f"bonds_{self.attr_name}"] = d[self.attr_name][data["atoms"]["num_nodes"]:data["monosacchs"]["num_nodes"]]
            data[f"monosacchs_{self.attr_name}"] = d[self.attr_name][data["monosacchs"]["num_nodes"]:]
        return data


class RandomWalkPE(RootTransform):
    """Random walk dense version."""
    attr_name = "rw_pe"

    def __init__(self, dim: int, individual: bool = True, cuda: bool = False, **kwargs: Any):
        """
        Args:
            dim: The number of random walk steps (walk_length).
            individual: Whether to compute eigenvectors for each node type individually.
            cuda: Whether to move the computation to GPU or not
            kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.walk_length = dim
        self.individual = individual
        self.cuda = cuda

    def forward(self, data: Data) -> Data:
        if data["num_nodes"] == 1:
            data[self.attr_name] = torch.zeros(1, 20)
            return data
        adj = to_dense_adj(data.edge_index, max_num_nodes=data["num_nodes"]).squeeze(0)
        row_sums = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sums.clamp(min=1)
        pe_list = [torch.zeros(adj.shape[0])]
        walk_matrix = adj
        for _ in range(self.walk_length - 1):
            walk_matrix = walk_matrix @ adj
            pe_list.append(walk_matrix.diag())
        pe = torch.stack(pe_list, dim=-1)
        data[self.attr_name] = pe
        return data

    def __call__(self, data):
        if self.individual:
            for d, name in zip(split_hetero_graph(data), ["atoms", "bonds", "monosacchs"]):
                self.forward(d)
                data[f"{name}_{self.attr_name}"] = d[self.attr_name]
        else:
            d = hetero_to_homo(data)
            self.forward(d)
            data[f"atoms_{self.attr_name}"] = d[self.attr_name][:data["atoms"]["num_nodes"]]
            data[f"bonds_{self.attr_name}"] = d[self.attr_name][data["atoms"]["num_nodes"]:data["monosacchs"]["num_nodes"]]
            data[f"monosacchs_{self.attr_name}"] = d[self.attr_name][data["monosacchs"]["num_nodes"]:]
        return data


class TQDMCompose(Compose):
    def forward(self, data: Union[Data, HeteroData]):
        # print(self.transforms)
        # print(len(self.transforms))
        # print(len(data))
        # with tqdm(total=len(self.transforms), desc="Transforms") as t_bar:
        for transform in tqdm(self.transforms, desc=f"Transform"):
            if not isinstance(data, (list, tuple)):
                data = transform(data)
            else:
                # data = [transform(d) for d in data]
                t_data = []
                for d in tqdm(data, leave=False):
                    t_data.append(transform(d))
                data = t_data
                    # s_bar.update(1)
                # data = [transform(d) for d in tqdm(data, total=len(data), desc="Samples", leave=False)]
                # t_bar.update(1)
        return data


def get_pretransforms(**pre_transform_args) -> [T.Compose]:
    pre_transforms = [
        GIFFLARTransform(**pre_transform_args.get("GIFFLARTransform", {})),
        GNNGLYTransform(**pre_transform_args.get("GNNGLYTransform", {})),
        ECFPTransform(**pre_transform_args.get("ECFPTransform", {})),
        SweetNetTransform(**pre_transform_args.get("SweetNetTransform", {})),
    ]
    print(pre_transform_args)
    for name, args in pre_transform_args.items():
        if name == "LaplacianPE":
            pre_transforms.append(LaplacianPE(**args))
        if name == "RandomWalkPE":
            pre_transforms.append(RandomWalkPE(**args))
    return TQDMCompose(pre_transforms)
