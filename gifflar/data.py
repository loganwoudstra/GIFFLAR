import copy
import pickle
from pathlib import Path
from typing import List, Union, Tuple, Optional, Callable, Any, Dict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, HeteroData
from pytorch_lightning import LightningDataModule
import glyles
from glyles.glycans.factory.factory import MonomerFactory
from tqdm import tqdm
import networkx as nx

from gifflar.utils import S3NMerger, nx2mol

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
RDLogger.DisableLog('rdApp.info')


def clean_tree(tree):
    for node in tree.nodes:
        attributes = copy.deepcopy(tree.nodes[node])
        if "type" in attributes and isinstance(attributes["type"], glyles.glycans.mono.monomer.Monomer):
            tree.nodes[node].clear()
            tree.nodes[node].update({"iupac": "".join([x[0] for x in attributes["type"].recipe]), "name": attributes["type"].name})
        else:
            return None
    return tree

def iupac2mol(iupac: str) -> Optional[HeteroData]:
    """
    Convert a glycan stored given as IUPAC-condensed string into an RDKit molecule while keeping the information of
    which atom and which bond belongs to which monosaccharide
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


# iupac2mol()


# GlyLES produces wrong structure?!, even wrong modifications?
# print(iupac2mol("GlcA(b1-2)ManOAc"))
# print(iupac2mol("GlcNAc(b1-3)LDManHep(a1-7)LDManHep(a1-3)[Glc(b1-4)]LDManHep(a1-5)[KoOPEtN(a2-4)]Kdo"))


class HeteroDataBatch:
    """Plain, dict-like object to store batches of HeteroDat-points"""

    def __init__(self, **kwargs: Any):
        """
        Initialize the object by setting each given argument as attribute of the object.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device: str):
        """
        Convert each field to the provided device by iteratively and recursively converting each field.

        Args:
            device: Name of device to convert to, e.g., CPU, cuda:0, ...

        Returns:
            Converted version of itself
        """
        for k, v in self.__dict__.items():
            if hasattr(v, "to"):
                setattr(self, k, v.to(device))
            elif isinstance(v, dict):
                for key, value in v.items():
                    if hasattr(value, "to"):
                        v[key] = value.to(device)
        return self

    def __getitem__(self, item: str) -> Any:
        """
        Get attribute from the object.

        Args:
            item: Name of the queries attribute

        Returns:
            THe queries attribute
        """
        return getattr(self, item)


def hetero_collate(data: Optional[Union[List[List[HeteroData]], List[HeteroData]]]) -> HeteroDataBatch:
    """
    Collate a list of HeteroData objects to a batch thereof.

    Args:
        data: The list of HeteroData objects

    Returns:
        A HeteroDataBatch object of the collated input samples
    """
    # If, for whatever reason the input is a list of lists, fix that
    if isinstance(data[0], list):
        data = data[0]

    # Extract all valid node types and edge types
    node_types = [t for t in data[0].node_types if len(data[0][t]) > 0]
    edge_types = data[0].edge_types

    # Setup empty fields for the most important attributes of the resulting batch
    x_dict = {}
    batch_dict = {}
    edge_index_dict = {}
    edge_attr_dict = {}

    # Include data for the baselines and other kwargs for house-keeping
    baselines = {"gnngly", "sweetnet"}
    kwargs = {key: [] for key in dict(data[0]) if all(b not in key for b in baselines)}

    # Store the node counts to offset edge indices when collating
    node_counts = {node_type: [0] for node_type in node_types}
    for d in data:
        for key in kwargs:
            # Collect all length-queryable fields
            if not hasattr(d[key], "__len__") or len(d[key]) != 0:
                kwargs[key].append(d[key])

        # Compute the offsets for each node type for sample identification after batching
        for node_type in node_types:
            node_counts[node_type].append(node_counts[node_type][-1] + d[node_type].num_nodes)

    # Collect the node features for each node type and store their assignment to the individual samples
    for node_type in node_types:
        x_dict[node_type] = torch.concat([d[node_type].x for d in data], dim=0)
        batch_dict[node_type] = torch.cat([
            torch.full((d[node_type].num_nodes,), i, dtype=torch.long) for i, d in enumerate(data)
        ], dim=0)

    # Collect edge information for each edge type
    for edge_type in edge_types:
        tmp_edge_index = []
        tmp_edge_attr = []
        for i, d in enumerate(data):
            # Collect the edge indices and offset them according to the offsets of their respective nodes
            if list(d[edge_type].edge_index.shape) == [0]:
                continue
            tmp_edge_index.append(torch.stack([
                d[edge_type].edge_index[0] + node_counts[edge_type[0]][i],
                d[edge_type].edge_index[1] + node_counts[edge_type[2]][i]
            ]))

            # Also collect edge attributes if existent (NOT TESTED!)
            if hasattr(d[edge_type], "edge_attr"):
                tmp_edge_attr.append(d[edge_type].edge_attr)

        # Collate the edge information
        edge_index_dict[edge_type] = torch.cat(tmp_edge_index, dim=1)
        if len(tmp_edge_attr) != 0:
            edge_attr_dict[edge_type] = torch.cat(tmp_edge_attr, dim=0)

    # For each baseline, collate it's node features and edge indices as well
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

    # Remove all incompletely given data and concat lists of tensors into single tensors
    num_nodes = {node_type: x_dict[node_type].shape[0] for node_type in node_types}
    for key, value in list(kwargs.items()):
        if any(key.startswith(b) for b in baselines):
            continue
        elif len(value) != len(data):
            del kwargs[key]
        elif isinstance(value[0], torch.Tensor):
            kwargs[key] = torch.cat(value, dim=0)

    # Finally create and return the HeteroDataBatch
    return HeteroDataBatch(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict,
                           num_nodes=num_nodes, batch_dict=batch_dict, **kwargs)


class GlycanStorage:
    def __init__(self, path: Optional[Path] = None):
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
            The IUPAC string of the query glycan

        Returns:
            A HeteroData object corresponding to the IUPAC string or None, if the IUPAC string could not be processed.
        """
        if iupac not in self.data:
            try:
                self.data[iupac] = iupac2mol(iupac)
            except:
                self.data[iupac] = None
        return copy.deepcopy(self.data[iupac])

    def _load(self) -> Dict[str, Optional[HeteroData]]:
        """
        Load the internal dictionary from the file, if it exists, otherwise, return an empty dict.

        Returns:
            The loaded (if possible) or an empty dictionary
        """
        if self.path.exists():
            with open(self.path, "rb") as f:
                return pickle.load(f)
        return {}


class GlycanDataModule(LightningDataModule):
    """DataModule holding datasets for Glycan-specific training"""
    def __init__(self, batch_size: int = 128, **kwargs: Any):
        """
        Initialize the DataModule with a given batch size.

        Args:
            batch_size: The batch size to use for training, validation, and testing
        """
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the training data.

        Returns:
            DataLoader for the training data
        """
        return DataLoader(self.train, batch_size=min(self.batch_size, len(self.train)), shuffle=True ,
                          collate_fn=hetero_collate)

    def val_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the validation data.

        Returns:
            DataLoader for the validation data
        """
        return DataLoader(self.val, batch_size=min(self.batch_size, len(self.val)), shuffle=False,
                          collate_fn=hetero_collate)

    def test_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the test data.

        Returns:
            DataLoader for the test data
        """
        return DataLoader(self.test, batch_size=min(self.batch_size, len(self.test)), shuffle=False,
                          collate_fn=hetero_collate)


class PretrainGDM(GlycanDataModule):
    """DataModule for pretraining a model on glycan data."""
    def __init__(
            self, root: str | Path,
            filename: str | Path,
            hash_code: str,
            batch_size: int = 64,
            train_frac: float = 0.8,
            **kwargs: Any,
    ):
        """
        Initialize the DataModule with the given parameters.

        Args:
            root: The root directory to store the processed data
            filename: The filename of the data to process
            hash_code: The hash code to use for the processed data
            batch_size: The batch size to use for training, validation, and testing
            train_frac: The fraction of the data to use for training
            **kwargs: Additional arguments to pass to the Pretrain
        """
        super().__init__(batch_size)
        ds = PretrainGDs(root=root, filename=filename, hash_code=hash_code, **kwargs)
        self.train, self.val = torch.utils.data.dataset.random_split(ds, [train_frac, 1 - train_frac])


class DownsteamGDM(GlycanDataModule):
    """DataModule for downstream tasks on glycan data."""
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            hash_code: str,
            batch_size: int = 64,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            **dataset_args
    ):
        """
        Initialize the DataModule with the given parameters.

        Args:
            root: The root directory to store the processed data
            filename: The filename of the data to process
            hash_code: The hash code to use for the processed data
            batch_size: The batch size to use for training, validation, and testing
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            **dataset_args: Additional arguments to pass to the DownstreamGDs
        """
        super().__init__(batch_size)
        self.train = DownstreamGDs(
            root=root, filename=filename, split="train", hash_code=hash_code, transform=transform,
            pre_transform=pre_transform, **dataset_args,
        )
        self.val = DownstreamGDs(
            root=root, filename=filename, split="val", hash_code=hash_code, transform=transform,
            pre_transform=pre_transform, **dataset_args,
        )
        self.test = DownstreamGDs(
            root=root, filename=filename, split="test", hash_code=hash_code, transform=transform,
            pre_transform=pre_transform, **dataset_args,
        )


class GlycanDataset(InMemoryDataset):
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            hash_code: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            path_idx: int = 0,
            **dataset_args,
    ):
        """
        Initialize the dataset with the given parameters.

        Args:
            root: The root directory to store the processed data
            filename: The filename of the data to process
            hash_code: The hash code to use for the processed data
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            path_idx: The index of the processed file name to use
            **dataset_args: Additional arguments to pass to the dataset
        """
        self.filename = Path(filename)
        self.dataset_args = dataset_args
        super().__init__(root=str(Path(root) / f"{filename.stem}_{hash_code}"), transform=transform,
                         pre_transform=pre_transform)
        self.data, self.dataset_args = torch.load(self.processed_paths[path_idx])

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.data.__len__()

    def len(self) -> int:
        """Return the length of the dataset."""
        return len(self)

    def __getitem__(self, item) -> Any:
        """Return the item at the given index."""
        return self.data[item]

    @property
    def processed_paths(self) -> List[str]:
        """Return the list of processed paths."""
        return [str(Path(self.root) / f) for f in self.processed_file_names]

    def process_(self, data, path_idx: int = 0) -> None:
        """Filter, process the data and store it at the given path index."""
        if self.pre_filter is not None:
            data = [d for d in data if self.pre_filter(d)]
        if self.pre_transform is not None:
            data = [self.pre_transform(d) for d in data]

        torch.save((data, self.dataset_args), self.processed_paths[path_idx])


class PretrainGDs(GlycanDataset):
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            hash_code: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            **dataset_args
    ):
        """
        Initialize the dataset for pre-training with the given parameters.

        Args:
            root: The root directory to store the processed data
            filename: The filename of the data to process
            hash_code: The hash code to use for the processed data
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            **dataset_args: Additional arguments to pass to the dataset
        """
        super().__init__(root=root, filename=filename, hash_code=hash_code, transform=transform,
                         pre_transform=pre_transform, **dataset_args)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        """Return the list of processed file names"""
        return [self.filename.stem + ".pt"]

    def process(self):
        """Process the data and store it."""
        data = []
        # to be implemented
        self.process_(data)


class DownstreamGDs(GlycanDataset):
    """Dataset for downstream tasks on glycan data."""
    splits = {"train": 0, "val": 1, "test": 2}

    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            split: str,
            hash_code: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            **dataset_args: Any,
    ):
        """
        Initialize the dataset for downstream tasks with the given parameters.

        Args:
            root: The root directory to store the processed data
            filename: The filename of the data to process
            split: The split to use, e.g., train, val, test
            hash_code: The hash code to use for the processed data
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            **dataset_args: Additional arguments to pass to the dataset
        """
        super().__init__(root=root, filename=filename, hash_code=hash_code, transform=transform,
                         pre_transform=pre_transform, path_idx=self.splits[split], **dataset_args)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        """Return the list of processed file names."""
        return [split + ".pt" for split in self.splits.keys()]

    def to_statistical_learning(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Convert the data to a format suitable for statistical learning (sklearn).

        Returns:
            The features, the labels, and the one-hot encoded labels
        """
        X, y, y_oh = [], [], []
        for d in self.data:
            X.append(d["fp"])
            y.append(d["y"])
            if hasattr(d, "y_oh"):
                y_oh.append(d["y_oh"])
        if isinstance(y[0], int):
            return np.vstack(X), np.array(y), np.vstack(y_oh) if len(y_oh) != 0 else None
        else:
            return np.vstack(X), np.concatenate(y), np.vstack(y_oh) if len(y_oh) != 0 else None

    def process(self) -> None:
        print("Start processing")
        """Process the data and store it."""
        data = {k: [] for k in self.splits}
        df = pd.read_csv(self.filename, sep="\t" if self.filename.suffix.lower().endswith(".tsv") else ",")

        # If the label is not given, use all columns except IUPAC and split
        if "label" not in self.dataset_args:
            self.dataset_args["label"] = [x for x in df.columns if x not in {"IUPAC", "split"}]
        if self.dataset_args["task"] != "classification":
            self.dataset_args["num_classes"] = len(self.dataset_args["label"])
        else:
            self.dataset_args["num_classes"] = int(max(df[self.dataset_args["label"]].values)) + 1
        print(self.dataset_args["num_classes"])
        if self.dataset_args["num_classes"] == 2:
            self.dataset_args["num_classes"] = 1
        # Load the glycan storage to speed up the preprocessing
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
