import json
import torch

from Code.Dataset.HypothesisDataset import HypothesisDataset
from Code.Models.RandoLMForest import prepare_inputs


class RFAMZWalkDataset:
    def __init__(self, walks, args: dict, walk_type: str):
        self.walks = walks
        self.walk_type = walk_type
        self.args = args
        self.feature_ordering = list(self.walks[0][0][-1].keys())

        self.inputs, self.targets, self.indices = prepare_inputs(
            data=self.walks,
            vocab_size=None,
            walk_type=self.walk_type,
            args=self.args,
        )


class AMZWalkDataset(torch.utils.data.Dataset):
    def __init__(self, walks, args: dict, walk_type: str, max_length: int = None):
        self.walks = walks
        self.walk_type = walk_type
        self.args = args
        self.feature_ordering = list(self.walks[0][0][-1].keys())
        self.max_length = max_length

    def __getitem__(self, index):
        annotated_walk = self.walks[index]
        if f"{self.walk_type}2id" in self.args:
            mapping = self.args[f"{self.walk_type}2id"]
        else:
            mapping = {i: i for i in range(self.args['size'])}  # identity mapping for subtrails (i know, programming could be better )
        if self.walk_type == "fkt":
            inDataPointIndex = 1
        elif self.walk_type == "kat":
            inDataPointIndex = 2
        elif "subtrails" in self.walk_type:
            inDataPointIndex = 1
        else:
            raise ValueError(f"Walk type {self.walk_type} not found")

        datapoint = torch.tensor([mapping[a[inDataPointIndex]] for a in annotated_walk], dtype=torch.int64)
        datapoint += 2  # for BOS and EOS
        datapoint = torch.cat([torch.tensor([0], dtype=torch.int64), datapoint, torch.tensor([1], dtype=torch.int64)])

        if self.max_length is not None:
            datapoint = torch.cat([datapoint, -torch.ones(self.max_length - len(datapoint), dtype=torch.int64)])

        return datapoint, {
            k: torch.tensor([v], dtype=torch.float32 if type(v) == float else torch.int64)
            for (k, v) in annotated_walk[0][-1].items()
        }

    def __len__(self):
        return len(self.walks)


class ReviewsDataset(HypothesisDataset):
    def __init__(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.__dict__.update(data)

    def get_walks(self):
        return self.annotated_walks, ...



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from pathlib import Path
    dataset_to_train = "data-subtrails-even"
    print("Loading dataset")
    dataset = ReviewsDataset(Path("data", "all-data", "dataset-subtrails.jsonl"))
    walks, _ = dataset.get_walks()
    print("Loading dataset done. Walks are loaded.")
    train_dataset = AMZWalkDataset(
        walks["-".join(dataset_to_train.split("-")[2:])],
        walk_type=dataset_to_train,
        args=dataset.args
    )
    for i in range(2):
        print(train_dataset[i])