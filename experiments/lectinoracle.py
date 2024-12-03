from tqdm import tqdm
import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
from glycowork.ml.model_training import train_model, SAM
from glycowork.ml.models import prep_model

from gifflar.data.modules import LGI_GDM
from gifflar.data.datasets import GlycanOnDiskDataset
from experiments.lgi_model import LectinStorage

le = LectinStorage("ESM", 33)

class LGI_OnDiskDataset(GlycanOnDiskDataset):
    @property
    def processed_file_names(self):
        """Return the list of processed file names."""
        return [split + ".db" for split in ["train", "val", "test"]]


def get_ds(dl, split_idx: int):
    ds = LGI_OnDiskDataset(root="/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/glycowork_data", path_idx=split_idx)
    data = []
    for x in tqdm(dl):
        data.append(Data(
            labels=x["sweetnet_x"],
            y=x["y"],
            edge_index=x["sweetnet_edge_index"],
            aa_seq=x["aa_seq"][0],
        ))
        if len(data) == 100:
            ds.extend(data)
            del data
            data = []
    if len(data) != 0:
        ds.extend(data)
        del data


def collate_lgi(data):
    for d in data:
        d["train_idx"] = le.query(d["aa_seq"])

    offset = 0
    labels, edges, y, train_idx, batch = [], [], [], [], []
    for i, d in enumerate(data):
        labels.append(d["labels"])
        edges.append(torch.stack([
            d["edge_index"][0] + offset,
            d["edge_index"][1] + offset,
        ]))
        offset += len(d["labels"])
        y.append(d["y"])
        train_idx.append(le.query(d["aa_seq"]))
        batch += [i for _ in range(len(d["labels"]))]

    labels = torch.cat(labels, dim=0)
    edges = torch.cat(edges, dim=1)
    y = torch.stack(y)
    train_idx = torch.stack(train_idx)
    batch = torch.tensor(batch)
    
    return Batch(
        labels=labels,
        edge_index=edges,
        y=y,
        train_idx=train_idx,
        batch=batch,
    )

datamodule = LGI_GDM(
    root="/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/lgi_data", filename="/home/rjo21/Desktop/GIFFLAR/lgi_data_20.pkl", hash_code="8b34af2a",
    batch_size=1, transform=None, pre_transform={"GIFFLARTransform": "", "SweetNetTransform": ""},
)

#get_ds(datamodule.train_dataloader(), 0)
#get_ds(datamodule.val_dataloader(), 1)
#get_ds(datamodule.test_dataloader(), 2)

train_set = LGI_OnDiskDataset("/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/glycowork_data", path_idx=0)
val_set = LGI_OnDiskDataset("/scratch/SCRATCH_SAS/roman/Gothenburg/GIFFLAR/glycowork_data", path_idx=1)

model = prep_model("LectinOracle", num_classes=1)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

m = train_model(
    model=model,
    dataloaders={"train": torch.utils.data.DataLoader(train_set, batch_size=128, collate_fn=collate_lgi), 
                 "val": torch.utils.data.DataLoader(val_set, batch_size=128, collate_fn=collate_lgi)},
    criterion=torch.nn.MSELoss(),
    optimizer=optimizer,
    scheduler=scheduler,
    return_metrics=True,
    mode="regression",
    num_epochs=100,
    patience=100,
)

import pickle

with open("lectinoracle_metrics.pkl", "wb") as f:
    pickle.dump(m, f)
