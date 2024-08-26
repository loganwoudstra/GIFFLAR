import copy
from typing import Any, Literal

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import Compose

from gifflar.pretransforms import RootTransform
from gifflar.utils import atom_map, bond_map, lib_map


class Masking(RootTransform):
    def __init__(self, cell: Literal["atoms", "bonds", "monosacch"], prob: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.cell = cell
        self.prob = prob

    def forward(self, data: HeteroData) -> HeteroData:
        data[f"{self.cell}_y"] = copy.deepcopy(data[self.cell]["x"])
        data[f"{self.cell}_mask"] = torch.rand(data[self.cell].num_nodes) < self.prob
        data[self.cell]["x"][data[f"{self.cell}_mask"]] = 0
        return data


def blub(cell):
    match(cell):
        case "atoms":
            return len(atom_map) + 1
        case "bonds":
            return len(bond_map) + 1
        case "monosacch":
            return len(lib_map) + 1
        case _:
            return 0


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
        match(args["name"]):
            case "TypeMasking":
                transforms.append(Masking(**args))
                task_list.append({
                    "num_classes": blub(args["cell"]),
                    "task": "classification",
                })

    return Compose(transforms), task_list
