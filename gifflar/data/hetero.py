from typing import Any, Optional, Union

import torch
from torch_geometric.data import HeteroData


class HeteroDataBatch:
    """Plain, dict-like object to store batches of HeteroDat-points"""

    def __init__(self, **kwargs: Any):
        """
        Initialize the object by setting each given argument as attribute of the object.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device: str) -> "HeteroDataBatch":
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


def determine_concat_dim(tensors):
    if len(tensors[0].shape) == 1:
        return 0
    concat_dim = None
    if all(tensor.shape[1] == tensors[0].shape[1] for tensor in tensors):
        concat_dim = 0  # Concatenate along rows (dim 0)
    elif all(tensor.shape[0] == tensors[0].shape[0] for tensor in tensors):
        concat_dim = 1  # Concatenate along columns (dim 1)
    return concat_dim


def hetero_collate(obj: object, data: Optional[Union[list[list[HeteroData]], list[HeteroData]]], *args, **kwargs) -> HeteroDataBatch:
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
    baselines = {"gnngly", "sweetnet", "rgcn"}
    kwargs = {key: [] for key in dict(data[0]) if all(b not in key for b in baselines)}
    #print("\n[", [d["y"] for d in data], "]")
    # Store the node counts to offset edge indices when collating
    node_counts = {node_type: [0] for node_type in node_types}
    for d in data:
        for key in kwargs:
            # Collect all length-queryable fields
            try:
                if not hasattr(d[key], "__len__") or len(d[key]) != 0:
                    kwargs[key].append(d[key])
                #print(key, "failing")
            except Exception as e:
                #print(e)
                pass

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
        if len(tmp_edge_index) != 0:
            edge_index_dict[edge_type] = torch.cat(tmp_edge_index, dim=1)
        if len(tmp_edge_attr) != 0:
            edge_attr_dict[edge_type] = torch.cat(tmp_edge_attr, dim=0)

    # For each baseline, collate its node features and edge indices as well
    for b in baselines:
        if not f"{b}_num_nodes" in data[0]:
            continue
        kwargs[f"{b}_x"] = torch.cat([d[f"{b}_x"] for d in data], dim=0)
        edges = []
        batch = []
        e_types = []
        n_types = []
        node_counts = 0
        for i, d in enumerate(data):
            edges.append(d[f"{b}_edge_index"] + node_counts)
            node_counts += d[f"{b}_num_nodes"]
            if b == "rgcn":
                e_types += d["rgcn_edge_type"]
                n_types += d["rgcn_node_type"]
            batch.append(torch.full((d[f"{b}_num_nodes"],), i, dtype=torch.long))
        kwargs[f"{b}_edge_index"] = torch.cat(edges, dim=1)
        kwargs[f"{b}_batch"] = torch.cat(batch, dim=0)
        if b == "rgcn":
            kwargs["rgcn_edge_type"] = torch.tensor(e_types)
            kwargs["rgcn_node_type"] = n_types
            if hasattr(data[0], "rgcn_rw_pe"):
                kwargs["rgcn_rw_pe"] = torch.cat([d["rgcn_rw_pe"] for d in data], dim=0)
            if hasattr(data[0], "rgcn_lap_pe"):
                kwargs["rgcn_lap_pe"] = torch.cat([d["rgcn_lap_pe"] for d in data], dim=0)

    # Remove all incompletely given data and concat lists of tensors into single tensors
    num_nodes = {node_type: x_dict[node_type].shape[0] for node_type in node_types}
    for key, value in list(kwargs.items()):
        if any(key.startswith(b) for b in baselines):
            continue
        elif len(value) != len(data):
            #print("\n-------------------------------\n", key, "|", len(value), "|", len(data), "\n-------------------------------\n")
            del kwargs[key]
        elif isinstance(value[0], torch.Tensor):
            dim = determine_concat_dim(value)
            if dim is None:
                raise ValueError(f"Tensors for key {key} cannot be concatenated.")
            kwargs[key] = torch.cat(value, dim=dim)
    #print(kwargs)

    # Finally create and return the HeteroDataBatch
    return HeteroDataBatch(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attr_dict=edge_attr_dict,
                           num_nodes=num_nodes, batch_dict=batch_dict, **kwargs)


def hetero_tuple_collate(obj: object, data: list[tuple[HeteroData, HeteroData]], *args, **kwargs: Any) -> tuple[HeteroDataBatch, HeteroDataBatch]:
    data, decoys = tuple(map(list, zip(*data)))
    return hetero_collate(obj, data), hetero_collate(obj, decoys)
