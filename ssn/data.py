import copy
import pickle
from pathlib import Path
from typing import List, Union, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, HeteroData
from pytorch_lightning import LightningDataModule
import glyles
from glyles.glycans.factory.factory import MonomerFactory
from tqdm import tqdm

from ssn.pretransforms import assemble_transforms
from ssn.utils import S3NMerger, nx2mol, bond_map, lib_map


def iupac2mol(iupac: str = "Fuc(a1-4)Gal(a1-4)Glc"):
    glycan = glyles.Glycan(iupac)
    # print(glycan.get_smiles())
    tree = glycan.parse_tree
    merged = S3NMerger(MonomerFactory()).merge(tree, glycan.root_orientation, glycan.start)
    mol = nx2mol(merged)

    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    Chem.WedgeMolBonds(mol, mol.GetConformer())

    data = HeteroData()
    data["IUPAC"] = iupac
    data["smiles"] = Chem.MolToSmiles(mol)
    if len(data["smiles"]) < 10:
        return None

    data["atoms"].x = torch.tensor([[
        atom.GetAtomicNum(),
        atom.GetChiralTag(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        len(atom.GetBonds()),
        atom.GetHybridization(),
    ] for atom in mol.GetAtoms()])
    data["atoms"].num_nodes = len(data["atoms"].x)
    data["bonds"].x = []
    data["atoms", "coboundary", "atoms"].edge_index = []
    data["atoms", "to", "bonds"].edge_index = []
    data["bonds", "to", "monosacchs"].edge_index = []
    for bond in mol.GetBonds():
        data["bonds"].x.append([
            bond_map.get(bond.GetBondType(), 0),
            bond.IsInRing(),
            bond.GetIsConjugated(),
            bond.GetBondDir(),
        ])
        data["atoms", "coboundary", "atoms"].edge_index += [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                                                            (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())]
        data["atoms", "to", "bonds"].edge_index += [(bond.GetBeginAtomIdx(), bond.GetIdx()),
                                                    (bond.GetEndAtomIdx(), bond.GetIdx())]
        data["bonds", "to", "monosacchs"].edge_index.append((bond.GetIdx(), bond.GetIntProp("mono_id")))
    data["bonds"].x = torch.tensor(data["bonds"].x)
    data["bonds"].num_nodes = len(data["bonds"].x)
    data["atoms", "coboundary", "atoms"].edge_index = torch.tensor(data["atoms", "coboundary", "atoms"].edge_index, dtype=torch.long).T
    data["atoms", "to", "bonds"].edge_index = torch.tensor(data["atoms", "to", "bonds"].edge_index, dtype=torch.long).T
    data["bonds", "to", "monosacchs"].edge_index = torch.tensor(data["bonds", "to", "monosacchs"].edge_index, dtype=torch.long).T
    data["bonds", "boundary", "bonds"].edge_index = torch.tensor(
        [(bond1.GetIdx(), bond2.GetIdx()) for atom in mol.GetAtoms() for bond1 in atom.GetBonds() for bond2 in
         atom.GetBonds() if bond1.GetIdx() != bond2.GetIdx()], dtype=torch.long).T
    # TODO: Adjust to glycan-informed molecule
    data["bonds", "coboundary", "bonds"].edge_index = torch.tensor(
        [(bond1, bond2) for ring in mol.GetRingInfo().BondRings() for bond1 in ring for bond2 in ring if
         bond1 != bond2], dtype=torch.long).T
    data["monosacchs"].x = torch.tensor([[
        lib_map.get(tree.nodes[node]["type"].name, len(lib_map)),
    ] for node in tree.nodes])
    data["monosacchs"].num_nodes = len(data["monosacchs"].x)
    data["monosacchs", "boundary", "monosacchs"].edge_index = []
    for a, b in tree.edges:
        data["monosacchs", "boundary", "monosacchs"].edge_index += [(a, b), (b, a)]
    data["monosacchs", "boundary", "monosacchs"].edge_index = torch.tensor(
        data["monosacchs", "boundary", "monosacchs"].edge_index, dtype=torch.long).T
    return data


# GlyLES produces wrong structure?!, even wrong modifications?
# print(iupac2mol("GlcA(b1-2)ManOAc"))
# print(iupac2mol("GlcNAc(b1-3)LDManHep(a1-7)LDManHep(a1-3)[Glc(b1-4)]LDManHep(a1-5)[KoOPEtN(a2-4)]Kdo"))


class HeteroDataBatch:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device):
        for k, v in self.__dict__.items():
            if hasattr(v, "to"):
                setattr(self, k, v.to(device))
            elif isinstance(v, dict):
                for key, value in v.items():
                    if hasattr(value, "to"):
                        value = value.to(device)
        return self

    def __getitem__(self, item):
        return getattr(self, item)


def hetero_collate(data):
    node_types = data[0].node_types
    edge_types = data[0].edge_types
    x_dict = {}
    batch_dict = {}
    edge_index_dict = {}
    edge_attr_dict = {}
    kwargs = {key: [] for key in dict(data[0])}
    node_counts = {node_type: [0] + [] for node_type in node_types}
    for d in data:
        for key in kwargs:
            if len(d[key]) != 0:
                kwargs[key].append(d[key])
        for node_type in node_types:
            node_counts[node_type].append(node_counts[node_type][-1] + d[node_type].num_nodes)

    for node_type in node_types:
        x_dict[node_type] = torch.concat([d[node_type].x for d in data], dim=0)
        batch_dict[node_type] = torch.cat([torch.full((d[node_type].num_nodes,), i, dtype=torch.long) for i, d in enumerate(data)], dim=0)

    for edge_type in edge_types:
        tmp_edge_index = []
        tmp_edge_attr = []
        for i, d in enumerate(data):
            tmp_edge_index.append(torch.stack([
                d[edge_type].edge_index[0] + node_counts[edge_type[0]][i],
                d[edge_type].edge_index[1] + node_counts[edge_type[2]][i]
            ]))
            if hasattr(d[edge_type], "edge_attr"):
                tmp_edge_attr.append(d[edge_type].edge_attr)
        edge_index_dict[edge_type] = torch.cat(tmp_edge_index, dim=1)
        if len(tmp_edge_attr) != 0:
            edge_attr_dict[edge_type] = torch.cat(tmp_edge_attr, dim=0)
    num_nodes = {node_type: x_dict[node_type].shape[0] for node_type in node_types}
    for key, value in list(kwargs.items()):
        if len(value) != len(data):
            del kwargs[key]
            continue
        if isinstance(value[0], torch.Tensor):
            kwargs[key] = torch.cat(value, dim=0)
    return HeteroDataBatch(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict,
                           num_nodes=num_nodes, batch_dict=batch_dict, **kwargs)


class GlycanStorage:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or Path("data")) / "glycan_storage.pkl"
        self.data = self._load()

    def close(self):
        with open(self.path, "wb") as out:
            pickle.dump(self.data, out)

    def query(self, iupac):
        if iupac not in self.data:
            self.data[iupac] = iupac2mol(iupac)
        return copy.deepcopy(self.data[iupac])

    def _load(self):
        if self.path.exists():
            with open(self.path, "rb") as f:
                return pickle.load(f)
        return {}


class GlycanDataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=min(self.batch_size, len(self.train)), shuffle=True, collate_fn=hetero_collate)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=min(self.batch_size, len(self.val)), shuffle=False, collate_fn=hetero_collate)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=min(self.batch_size, len(self.test)), shuffle=False, collate_fn=hetero_collate)


class PretrainGDM(GlycanDataModule):
    def __init__(self, root: str | Path, filename: str | Path, batch_size: int = 64, train_frac: float = 0.8):
        super().__init__(batch_size)
        ds = PretrainGDs(root=root, filename=filename)
        self.train, self.val = torch.utils.data.dataset.random_split(ds, [train_frac, 1 - train_frac])


class DownsteamGDM(GlycanDataModule):
    def __init__(self, root, filename, batch_size, **kwargs):
        super().__init__(batch_size)
        pre_transform, transform = assemble_transforms(**kwargs)
        print(pre_transform, "\n", transform)
        self.train = DownstreamGDs(root=root, filename=filename, split="train", name=kwargs["model"]["name"],
                                   transform=transform, pre_transform=pre_transform)
        self.val = DownstreamGDs(root=root, filename=filename, split="val",  name=kwargs["model"]["name"],
                                 transform=transform, pre_transform=pre_transform)
        self.test = DownstreamGDs(root=root, filename=filename, split="test",  name=kwargs["model"]["name"],
                                  transform=transform, pre_transform=pre_transform)


class GlycanDataset(InMemoryDataset):
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            name: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            path_idx: int = 0,
    ):
        self.filename = Path(filename)
        super().__init__(root=str(Path(root) / filename.stem / name), transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[path_idx])

    @property
    def processed_paths(self) -> List[str]:
        return [str(Path(self.root) / f) for f in self.processed_file_names]

    def process_(self, data, path_idx: int = 0):
        if self.pre_filter is not None:
            data = [d for d in data if self.pre_filter(d)]
        if self.pre_transform is not None:
            data = [self.pre_transform(d) for d in data]

        data, slices = self.collate(data)
        torch.save((data, slices), self.processed_paths[path_idx])


class PretrainGDs(GlycanDataset):
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            name: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None
    ):
        super().__init__(root=root, filename=filename, name=name, transform=transform, pre_transform=pre_transform)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return [self.filename.stem + ".pt"]

    def process(self):
        data = []
        # to be implemented
        self.process_(data)


class DownstreamGDs(GlycanDataset):
    splits = {"train": 0, "val": 1, "test": 2}

    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            split: str,
            name: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
    ):
        if name in ["rf", "svm", "xgb"]:
            name = "mlp"
        super().__init__(root=root, filename=filename, name=name, transform=transform, pre_transform=pre_transform,
                         path_idx=self.splits[split])

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return [split + ".pt" for split in self.splits.keys()]

    def to_statistical_learning(self):
        X, y, y_oh = [], [], []
        for d in self:
            X.append(d["fp"])
            y.append(d["y"])
            if "y_oh" in d:
                y_oh.append(d["y_oh"])
        return np.vstack(X), np.concatenate(y), np.vstack(y_oh) if len(y_oh) != 0 else None

    def process(self):
        data = {k: [] for k in self.splits}
        df = pd.read_csv(self.filename, sep="\t")
        gs = GlycanStorage(self.root)
        for index, (_, row) in tqdm(enumerate(df.iterrows())):
            try:
                d = gs.query(row["glycan"])
                if d is None:
                    continue
                if isinstance(row["label"], str) and "[" in row["label"] and "]" in row["label"]:
                    d["y_oh"] = torch.tensor([int(x) for x in row["label"][1:-1].split(" ")])
                    d["y"] = d["y_oh"].argmax().item()
                else:
                    d["y"] = row["label"]
                d["ID"] = index
                data[row["split"]].append(d)
            except Exception as e:
                print("Failed to process", row["glycan"], "\n\t", e)
        gs.close()
        print("Processed", sum(len(v) for v in data.values()), "entries")
        for split in self.splits:
            self.process_(data[split], path_idx=self.splits[split])
