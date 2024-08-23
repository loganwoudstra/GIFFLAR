import copy

import torch
from torch_geometric.transforms import Compose

from gifflar.pretransforms import RootTransform
from gifflar.utils import atom_map, bond_map, lib_map


class MonosaccharideMasking(RootTransform):
    def __call__(self, data):
        pass


class Masking(RootTransform):
    def __init__(self, cell, prob, **kwargs):
        super().__init__(**kwargs)
        self.cell = cell
        self.prob = prob

    def forward(self, data):
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


def get_transforms(**transform_args) -> tuple[Compose, list[dict]]:
    """
    Get the transforms for the training.

    Params:
        transform_args: The arguments for the transforms.

    Returns:
        transforms: The transforms for the training.
    """
    transforms = []
    task_list = []
    for name, args in transform_args.items():
        if name == "TypeMasking":
            transforms.append(Masking(**args))
            task_list.append({
                "num_classes": blub(args["cell"]),
                "task": "classification",
            })
    return Compose(transforms), task_list