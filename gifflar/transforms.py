import copy
from typing import Any, Literal

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import Compose

from gifflar.pretransforms import RootTransform
from gifflar.utils import get_number_of_classes


class Masking(RootTransform):
    """Mask a random fraction of nodes in the graph."""

    def __init__(self, cell: Literal["atoms", "bonds", "monosacch"], prob: float, **kwargs: Any):
        """
        Mask a random fraction of nodes in the graph.

        Params:
            cell: The cell to mask.
            prob: The probability of masking a node.
            kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.cell = cell
        self.prob = prob

    def forward(self, data: HeteroData) -> HeteroData:
        """
        Mask a random fraction of nodes in the graph.

        Params:
            data: The data to transform.

        Returns:
            The transformed data.
        """
        data[f"{self.cell}_y"] = copy.deepcopy(data[self.cell]["x"])
        data[f"{self.cell}_mask"] = torch.rand(data[self.cell].num_nodes) < self.prob
        data[self.cell]["x"][data[f"{self.cell}_mask"]] = 0
        return data


def get_transforms(transform_args: list[dict[str, Any]]) -> tuple[Compose, list[dict]]:
    """
    Get the transforms for the training.

    Params:
        transform_args: The arguments for the transforms.

    Returns:
        transforms: The transforms for the training.
    """
    transforms = []
    task_list = []
    for args in transform_args:
        match (args["name"]):
            case "TypeMasking":
                transforms.append(Masking(**args))
                task_list.append({
                    "num_classes": get_number_of_classes(args["cell"]),
                    "task": "classification",
                })

    return Compose(transforms), task_list
