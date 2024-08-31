from pathlib import Path
from typing import Optional, Callable, Any

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from gifflar.data.datasets import DownstreamGDs, PretrainGDs
from gifflar.data.hetero import hetero_collate


class GlycanDataModule(LightningDataModule):
    """DataModule holding datasets for Glycan-specific training"""

    def __init__(self, batch_size: int = 128, num_workers: int = 16, **kwargs: Any):
        """
        Initialize the DataModule with a given batch size.

        Args:
            batch_size: The batch size to use for training, validation, and testing
            num_workers: The number of CPUs to use for loading the data
            **kwargs: Additional arguments to pass to the Data
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the training data.

        Returns:
            DataLoader for the training data
        """
        return DataLoader(self.train, batch_size=min(self.batch_size, len(self.train)), shuffle=True,
                          collate_fn=hetero_collate, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the validation data.

        Returns:
            DataLoader for the validation data
        """
        return DataLoader(self.val, batch_size=min(self.batch_size, len(self.val)), shuffle=False,
                          collate_fn=hetero_collate, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the test data.

        Returns:
            DataLoader for the test data
        """
        return DataLoader(self.test, batch_size=min(self.batch_size, len(self.test)), shuffle=False,
                          collate_fn=hetero_collate, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        """
        Combines the train, val, and test datasets and return the DataLoader for that data.

        Returns:
            DataLoader for the combined data
        """
        predict = ConcatDataset([self.train, self.val, self.test])
        self.batch_size = 4
        print("Batch-Size:", min(self.batch_size, len(predict)))
        return DataLoader(predict, batch_size=min(self.batch_size, len(predict)), shuffle=False,
                          collate_fn=hetero_collate, num_workers=self.num_workers)


class PretrainGDM(GlycanDataModule):
    """DataModule for pretraining a model on glycan data."""

    def __init__(
            self,
            file_path: str | Path,
            hash_code: str,
            batch_size: int = 64,
            train_frac: float = 0.8,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
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
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            **kwargs: Additional arguments to pass to the Pretrain
        """
        root = Path(file_path).parent
        super().__init__(batch_size)
        ds = PretrainGDs(root=root, filename=file_path, hash_code=hash_code, transform=transform,
                         pre_transform=pre_transform, **kwargs)
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
            **dataset_args: dict[str, Any],
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
