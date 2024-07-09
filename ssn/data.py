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

from ssn.utils import S3NMerger, nx2mol

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


def iupac2mol(iupac: str = "Fuc(a1-4)Gal(a1-4)Glc"):
    glycan = glyles.Glycan(iupac)
    # print(glycan.get_smiles())
    tree = glycan.parse_tree
    merged = S3NMerger(MonomerFactory()).merge(tree, glycan.root_orientation, glycan.start)
    mol = nx2mol(merged)

    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    Chem.WedgeMolBonds(mol, mol.GetConformer())

    smiles = Chem.MolToSmiles(mol)
    if len(smiles) < 10:
        return None

    data = HeteroData()
    data["IUPAC"] = iupac
    data["smiles"] = smiles
    data["mol"] = mol
    data["tree"] = tree
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
                        v[key] = value.to(device)
        return self

    def __getitem__(self, item):
        return getattr(self, item)


def hetero_collate(data):
    if isinstance(data[0], list):
        data = data[0]
    # Samples -> Batch
    node_types = [t for t in data[0].node_types if len(data[0][t]) > 0]
    edge_types = data[0].edge_types
    x_dict = {}
    batch_dict = {}
    edge_index_dict = {}
    edge_attr_dict = {}
    baselines = {"gnngly", "sweetnet"}
    kwargs = {key: [] for key in dict(data[0]) if all(b not in key for b in baselines)}
    node_counts = {node_type: [0] for node_type in node_types}
    for d in data:
        for key in kwargs:
            # The following does not work, because NodeStorage reports to not have len, but one can run len(NodeStorage)
            if not hasattr(d[key], "__len__") or len(d[key]) != 0:
                # if getattr(d[key], "len", 0) != 0:
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

    for b in baselines:
        kwargs[f"{b}_x"] = torch.cat([d[f"{b}_x"] for d in data], dim=0)
        edges = []
        batch = []
        node_counts = 0
        for i, d in enumerate(data):
            edges.append(d[f"{b}_edge_index"] + node_counts)
            node_counts += d[f"{b}_num_nodes"]
            batch.append(torch.full((d[f"{b}_num_nodes"],), i, dtype=torch.long))
        kwargs[f"{b}_edge_index"] = torch.cat(edges, dim=1)
        kwargs[f"{b}_batch"] = torch.cat(batch, dim=0)

    num_nodes = {node_type: x_dict[node_type].shape[0] for node_type in node_types}
    for key, value in list(kwargs.items()):
        if any(key.startswith(b) for b in baselines):
            continue
        elif len(value) != len(data):
            del kwargs[key]
        elif isinstance(value[0], torch.Tensor):
            kwargs[key] = torch.cat(value, dim=0)
    return HeteroDataBatch(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict,
                           num_nodes=num_nodes, batch_dict=batch_dict, **kwargs)


class GlycanStorage:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or "data") / "glycan_storage.pkl"
        self.data = self._load()

    def close(self):
        with open(self.path, "wb") as out:
            pickle.dump(self.data, out)

    def query(self, iupac):
        if iupac not in self.data:
            try:
                self.data[iupac] = iupac2mol(iupac)
            except:
                self.data[iupac] = None
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
        return DataLoader(self.train, batch_size=min(self.batch_size, len(self.train)), shuffle=True , collate_fn=hetero_collate)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=min(self.batch_size, len(self.val)), shuffle=False, collate_fn=hetero_collate)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=min(self.batch_size, len(self.test)), shuffle=False, collate_fn=hetero_collate)


class PretrainGDM(GlycanDataModule):
    def __init__(self, root: str | Path, filename: str | Path, batch_size: int = 64, train_frac: float = 0.8, **kwargs):
        super().__init__(batch_size)
        ds = PretrainGDs(root=root, filename=filename, **kwargs)
        self.train, self.val = torch.utils.data.dataset.random_split(ds, [train_frac, 1 - train_frac])


class DownsteamGDM(GlycanDataModule):
    def __init__(self, root, filename, batch_size, transform, pre_transform, **dataset_args):
        super().__init__(batch_size)
        self.train = DownstreamGDs(
            root=root, filename=filename, split="train", transform=transform, pre_transform=pre_transform, **dataset_args
        )
        self.val = DownstreamGDs(
            root=root, filename=filename, split="val", transform=transform, pre_transform=pre_transform, **dataset_args
        )
        self.test = DownstreamGDs(
            root=root, filename=filename, split="test", transform=transform, pre_transform=pre_transform, **dataset_args
        )


class GlycanDataset(InMemoryDataset):
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            path_idx: int = 0,
            **dataset_args,
    ):
        # Removed slices from collate, saving and loading. Might be unnecessary?
        self.filename = Path(filename)
        self.dataset_args = dataset_args
        super().__init__(root=str(Path(root) / filename.stem), transform=transform, pre_transform=pre_transform)
        self.data = torch.load(self.processed_paths[path_idx])

    def __len__(self):
        return self.data.__len__()

    def len(self):
        return len(self)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def processed_paths(self) -> List[str]:
        return [str(Path(self.root) / f) for f in self.processed_file_names]

    def process_(self, data, path_idx: int = 0):
        if self.pre_filter is not None:
            data = [d for d in data if self.pre_filter(d)]
        if self.pre_transform is not None:
            data = [self.pre_transform(d) for d in data]

        torch.save(data, self.processed_paths[path_idx])


class PretrainGDs(GlycanDataset):
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            **dataset_args
    ):
        super().__init__(root=root, filename=filename, transform=transform, pre_transform=pre_transform, **dataset_args)

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
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            **dataset_args
    ):
        super().__init__(root=root, filename=filename, transform=transform, pre_transform=pre_transform,
                         path_idx=self.splits[split], **dataset_args)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return [split + ".pt" for split in self.splits.keys()]

    def to_statistical_learning(self):
        X, y, y_oh = [], [], []
        for d in self.data:
            X.append(d["fp"])
            y.append(d["y"])
            if hasattr(d, "y_oh"):
                y_oh.append(d["y_oh"])
        return np.vstack(X), np.concatenate(y), np.vstack(y_oh) if len(y_oh) != 0 else None

    def process(self):
        data = {k: [] for k in self.splits}
        df = pd.read_csv(self.filename, sep="\t" if self.filename.suffix.lower().endswith(".tsv") else ",")
        gs = GlycanStorage(Path(self.root).parent)
        for index, (_, row) in tqdm(enumerate(df.iterrows())):
            d = gs.query(row["IUPAC"])
            if d is None:
                continue
            if self.dataset_args["task"] == "regression" or len(self.dataset_args["label"]) == 1:
                d["y"] = torch.tensor(list(row[self.dataset_args["label"]].values)).reshape(1, -1)
            elif len(self.dataset_args["label"]) > 1:
                d["y_oh"] = torch.tensor([int(x) for x in row[self.dataset_args["label"]]]).reshape(1, -1)
                if self.dataset_args["task"] != "multilabel":
                    d["y"] = d["y_oh"].argmax().item()
            d["ID"] = index
            data[row["split"]].append(d)
        gs.close()
        print("Processed", sum(len(v) for v in data.values()), "entries")
        for split in self.splits:
            self.process_(data[split], path_idx=self.splits[split])
