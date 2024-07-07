from typing import Tuple, Literal, Optional

import torch
from glycowork.glycan_data.loader import lib
from glycowork.motif.graph import glycan_to_nxGraph
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from torch import nn
from torch_geometric import transforms as T
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.utils import from_networkx

from ssn.utils import atom_map, bond_map


class RootTransform(BaseTransform):
    def __init__(self, **kwargs):
        pass


class ScalarTransform(RootTransform):
    # + 1 for unknown, + 1 for masking
    class_map = {
        "atoms": len(atom_map) + 2,
        "bonds": len(bond_map) + 2,
        "monosacchs": len(lib) + 2,
    }

    def __init__(
            self,
            level: str | Tuple[str, str, str],
            embed_dim: int,
            index: int,
            mode: Literal["random", "onehot"] = "random",
            num_classes: int = -1,
            **kwargs
    ):
        super().__init__()
        self.level = level
        self.node_mode = isinstance(self.level, str)
        self.index = index
        self.suffix = "_" + mode
        if num_classes == -1 and level in self.class_map:
            num_classes = self.class_map[self.level]
        else:
            raise ValueError(f"num_classes must be provided for {level}.")
        self.embedding = nn.Embedding(num_classes, embed_dim) if mode == "random" else torch.eye(num_classes)

    def forward(self, data):
        if self.node_mode:
            data[self.level]["x" + self.suffix] = self.embedding(data[self.level]["x"][:, self.index])
        else:
            data[self.level]["edge_attr" + self.suffix] = self.embedding(data[self.level]["edge_attr"][:, self.index])
        return data


class SimplexCleanTransform(RootTransform):
    def forward(self, data):
        data["atoms"]["x"] = torch.tensor(
            [atom_map.get(v.item(), len(atom_map) + 1) for v in data["atoms"]["x"][:, 0]]).reshape(-1, 1)
        return data


class GNNGLYTransform(RootTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.atom_type_embed = torch.eye(101)
        self.chiral_embed = torch.eye(4)
        self.degree_embed = torch.eye(13)
        self.formal_charge_embed = torch.eye(5)
        self.hydrogen_count_embed = torch.eye(5)
        self.hybridization_embed = torch.eye(5)

    def __call__(self, data):
        data["atoms"]["x"][:, 0] = data["atoms"]["x"][:, 0].clamp(max=100)
        atom_type_embed = self.atom_type_embed[data["atoms"]["x"][:, 0]]
        chiral_embed = torch.stack([self.chiral_embed[chiral] for chiral in data["atoms"]["x"][:, 1]])
        degree_embed = self.degree_embed[data["atoms"]["x"][:, 2]]
        formal_charge_embed = self.formal_charge_embed[data["atoms"]["x"][:, 3]]
        hydrogen_count_embed = self.hydrogen_count_embed[data["atoms"]["x"][:, 4]]
        hybridization_embed = torch.stack([self.hybridization_embed[hybrid] for hybrid in data["atoms"]["x"][:, 5]])
        data["atoms"]["x"] = torch.cat([
            atom_type_embed,
            chiral_embed,
            degree_embed,
            formal_charge_embed,
            hydrogen_count_embed,
            hybridization_embed,
        ], dim=1).float()
        return data


class MLPTransform(RootTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fpler = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    def __call__(self, data):
        data["fp"] = torch.tensor(self.fpler.GetFingerprint(Chem.MolFromSmiles(data["smiles"])), dtype=torch.float).reshape(1, -1)
        return data


class SweetNetTransform(RootTransform):
    def __call__(self, data):
        try:
            d = from_networkx(glycan_to_nxGraph(data["IUPAC"], lib))
        except:
            d = from_networkx(glycan_to_nxGraph("Gal", lib))
        data["atoms"]["x"] = torch.tensor(d["labels"]).reshape(-1, 1)
        data["atoms"]["num_nodes"] = len(data["atoms"]["x"])
        data["atoms", "coboundary", "atoms"]["edge_index"] = d.edge_index
        return data


class CollectionTransform(RootTransform):
    def forward(self, data):
        for type_ in data.node_types:
            attr = [data[type_][key] for key in list(data[type_].keys()) if key.startswith("x_")]
            if len(attr) != 0:
                data[type_]["x"] = torch.concat(attr, dim=1)
        for type_ in data.edge_types:
            attr = [data[type_][key] for key in list(data[type_].keys()) if key.startswith("edge_attr_")]
            if len(attr) != 0:
                data[type_]["edge_attr"] = torch.concat(attr, dim=1)
        return data


transformation_list = {
    "clean": SimplexCleanTransform,
    "scalar": ScalarTransform,
    "gnngly": GNNGLYTransform,
    "mlp": MLPTransform,
    "sweetnet": SweetNetTransform,
}


def assemble_transforms(**kwargs) -> Tuple[Optional[T.Compose], Optional[T.Compose]]:
    pre_transforms = T.Compose([
        transformation_list[args["name"]](**args) for args in kwargs["model"]["featurization"]
    ] + [CollectionTransform()])
    transforms = None
    return pre_transforms, transforms
