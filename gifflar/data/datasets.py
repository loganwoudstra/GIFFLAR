import pickle
from pathlib import Path
from typing import Union, Optional, Callable, Any
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, HeteroData, OnDiskDataset, Database, SQLiteDatabase
from torch_geometric.data.data import BaseData
from torch_geometric.data.database import Schema
from torch_geometric.data.dataset import Dataset
from tqdm import tqdm

from gifflar.data.utils import GlycanStorage


class GlycanOnDiskDataset(OnDiskDataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            schema: Schema = object,
            path_idx: int = 0,
            log: bool = True,
    ) -> None:
        self.backend = "sqlite"  # "sqlite" or "rocksdb"
        self.schema = schema
        self.path_idx = path_idx

        self._db: Optional[Database] = None
        self._numel: Optional[int] = None
        super(OnDiskDataset, self).__init__(root, transform, pre_transform, pre_filter, log=log)

    def get_db(self, path_idx):
        kwargs = {}
        cls = self.BACKENDS[self.backend]
        if issubclass(cls, SQLiteDatabase):
            kwargs['name'] = self.__class__.__name__

        os.makedirs(self.processed_dir, exist_ok=True)
        path = self.processed_paths[path_idx]
        db = cls(path=path, schema=self.schema, **kwargs)
        return db

    @property
    def db(self) -> Database:
        r"""Returns the underlying :class:`Database`."""
        if self._db is None:
            self._db = self.get_db(self.path_idx)
            self._numel = len(self._db)
        return self._db

    def get(self, idx: int) -> BaseData:
        d = super().get(idx)
        if self.transform is not None:
            d = self.transform(d)
        return d

    def __getitem__(self, idx: int):
        return self.get(idx)

    def process_(self, data: list[HeteroData], path_idx: int):
        """

        """
        db = self.get_db(path_idx)
        db.multi_insert(range(len(data)), data, batch_size=None)
        db.close()


class GlycanInMemoryDataset(InMemoryDataset):
    def __init__(
            self,
            root: str | Path,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            path_idx: int = 0,
    ) -> None:
        super().__init__(root, transform, pre_transform)
        self.data, self.dataset_args = torch.load(self.processed_paths[path_idx])
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.data.__len__()
    
    def len(self) -> int:
        """Return the length of the dataset."""
        return len(self)
    
    def __getitem__(self, item) -> Any:
        """Return the item at the given index."""
        return self.data[item] if self.transform is None else self.transform(self.data[item])

    def process_(self, data: list[HeteroData], path_idx: Path | str):
        torch.save((data, self.dataset_args), self.processed_paths[path_idx])


class GlycanDataset(GlycanOnDiskDataset):
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            hash_code: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            path_idx: int = 0,
            **dataset_args: dict[str, Any],
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
        super().__init__(root=str(Path(root) / f"{self.filename.stem}_{hash_code}"), transform=transform, pre_transform=pre_transform, path_idx=path_idx)

    @property
    def processed_paths(self) -> list[str]:
        """Return the list of processed paths."""
        return [str(Path(self.root) / f) for f in self.processed_file_names]

    def process_(self, data: list[HeteroData], path_idx: int = 0) -> None:
        """
        Filter, process the data and store it at the given path index.

        Args:
            data: The data to process
            path_idx: The index of the processed file name to use
        """
        if self.pre_filter is not None:
            data = [d for d in data if self.pre_filter(d)]
        if self.pre_transform is not None:
            print("Pre-Transform")
            data = self.pre_transform(data)

        super().process_(data, path_idx)


class PretrainGDs(GlycanDataset):
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            hash_code: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            **dataset_args: dict[str, Any],
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
    def processed_file_names(self) -> Union[str, list[str], tuple[str, ...]]:
        """Return the list of processed file names"""
        return [self.filename.stem + ".pt"]

    def process(self) -> None:
        """Process the data and store it."""
        data = []
        gs = GlycanStorage(Path(self.root).parent)
        # df = pd.read_csv(self.filename, sep="\t" if self.filename.suffix.lower().endswith(".tsv") else ",")
        with open(self.filename, "r") as glycans:
            for line in glycans.readlines():
                d = gs.query(line.strip())
                # d["ID"] = index
                data.append(d)
        gs.close()
        print("Processed", len(data), "entries")
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
            **dataset_args: dict[str, Any],
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
    def processed_file_names(self) -> Union[str, list[str], tuple[str, ...]]:
        """Return the list of processed file names."""
        return [split + ".pt" for split in self.splits.keys()]

    def to_statistical_learning(self) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
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
        """Process the data and store it."""
        print("Start processing")
        data = {k: [] for k in self.splits}
        df = pd.read_csv(self.filename, sep="\t" if self.filename.suffix.lower().endswith(".tsv") else ",")

        # If the label is not given, use all columns except IUPAC and split
        if "label" not in self.dataset_args:
            self.dataset_args["label"] = [x for x in df.columns if x not in {"IUPAC", "split"}]

        # Compute the number of classes
        if self.dataset_args["task"] != "classification":
            self.dataset_args["num_classes"] = len(self.dataset_args["label"])
        else:
            self.dataset_args["num_classes"] = int(max(df[self.dataset_args["label"]].values)) + 1
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


class LGIDataset(DownstreamGDs):
    def process(self) -> None:
        """Process the data and store it."""
        print("Start processing")
        print(self.pre_transform)
        data = {k: [] for k in self.splits}
        with open(self.filename, "rb") as f:
            inter, lectin_map, glycan_map = pickle.load(f)

        # Load the glycan storage to speed up the preprocessing
        gs = GlycanStorage(Path(self.root).parent)
        for i, (lectin_id, glycan_id, value, split) in tqdm(enumerate(inter)):
            d = gs.query(glycan_map[glycan_id])
            if d is None:
                continue
            d["aa_seq"] = lectin_map[lectin_id]
            d["y"] = torch.tensor([value])
            d["ID"] = i
            data[split].append(d)

        gs.close()
        print("Processed", sum(len(v) for v in data.values()), "entries")
        for split in self.splits:
            print("Post-process", split)
            self.process_(data[split], path_idx=self.splits[split])
