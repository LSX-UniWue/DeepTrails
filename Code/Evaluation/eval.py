import networkx as nx
import numpy as np
import torch


class Eval:
    def __init__(self, name: str, classes: int, graph_struct: nx.Graph) -> None:
        self.name = name
        self.classes = classes
        self.graph = graph_struct
        self.loss = 0.0
        self.ranks = []
        self.sequences = []
        self.predictions = []
        self.n = 0

    def __call__(self, loss: float, sequences: torch.tensor, predictions: torch.tensor) -> None:
        self.loss += loss
        self.n += len(predictions)
        self.sequences.extend(sequences.tolist())
        self.predictions.extend(predictions.tolist())
        ranks = (predictions == sequences[:, -1:].reshape(-1, 1)).nonzero()[:, -1].float()
        self.ranks.extend(ranks.tolist())

    def evaluate(self) -> dict:
        eval_dict = {}
        ranks = np.array(self.ranks)
        predictions = np.array(self.predictions)
        eval_dict["name"] = self.name
        eval_dict["n"] = self.n
        if isinstance(self.loss, torch.Tensor):
            eval_dict["loss"] = self.loss.item() / self.n
        else:
            eval_dict["loss"] = self.loss / self.n
        eval_dict["ranks"] = self.ranks
        eval_dict["acc"] = (ranks == 0).sum() / self.n
        eval_dict["hit@10"] = (ranks < 10).sum() / self.n
        eval_dict["ndcg"] = (1 / np.log2(2 + ranks)).sum() / self.n
        # count the amount times, the network predicted actual network neighbors
        curr_neighbors = [list(self.graph.neighbors(seq[-2])) for seq in self.sequences]
        eval_dict["neigh"] = sum([a in b for a, b in zip(predictions[:, 0], curr_neighbors)]) / self.n
        # count the amount of times, predictions were made for even network neighbors
        eval_dict["even"] = sum([a in b for a, b in zip(predictions[:, 0], curr_neighbors) if a % 2 == 0]) / self.n
        # count the amount of times, predictions were made for odd network neighbors
        eval_dict["odd"] = sum([a in b for a, b in zip(predictions[:, 0], curr_neighbors) if a % 2 == 1]) / self.n
        return eval_dict

    def reset(self):
        self.ranks = []
        self.sequences = []
        self.loss = 0.0
        self.n = 0
