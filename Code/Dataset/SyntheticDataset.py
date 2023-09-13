import random

from Code import MIXTURES
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from Code.Dataset.HypothesisDataset import HypothesisDataset


class WalkDataset(torch.utils.data.Dataset):
    def __init__(self, walks):
        self.walks = walks

    def __getitem__(self, index):
        datapoint = torch.tensor(self.walks[index], dtype=torch.int64)
        datapoint += 2  # for BOS and EOS
        datapoint = torch.cat([torch.tensor([0], dtype=torch.int64), datapoint, torch.tensor([1], dtype=torch.int64)])
        return datapoint

    def __len__(self):
        return len(self.walks)


class RFWalkDataset:
    def __init__(self, walks):
        self.walks = walks
        from Code.Models.RandoLMForest import prepare_inputs

        self.inputs, self.targets, self.indices = prepare_inputs(self.walks, vocab_size=max(max(walks)) + 1)


class SyntheticDataset(HypothesisDataset):
    def __init__(self, args: dict):
        self.args = args
        self.size = args["size"]
        self.connectivity = args["connectivity"]
        self.amount_walker = args["amount_walker"]
        self.walk_length = args["walk_length"]
        if "categories" in args:
            print("\n\tCreating Subtrails dataset\n")
            self.categories = args["categories"]
            self.walks_per_category = args["walks_per_category"]
            self.isSubtrails = True
            self.isMixedtrails = False
        elif "mixture" in args:
            print("\n\tCreating MixedTrails dataset\n")
            self.mixture = args["mixture"]
            self.isMixedtrails = True
            self.isSubtrails = False
        else:
            print("\n\tCreating HypTrais dataset\n")
            self.isMixedtrails = False
            self.isSubtrails = False
        self.verbose = args["verbose"]
        if "barabasi" in args["graph_type"]:
            self.graph = nx.barabasi_albert_graph(self.size, self.connectivity)
        else:  # will be complete graph
            self.graph = nx.complete_graph(self.size)
        if self.verbose:
            print("\n\tGraph generated\n")
        if self.isSubtrails:
            self.annotated_walks = {"annotated_walks": self.build_subtrails_walks(self.walks_per_category)}
        else:
            self.hypothesis_walks = {f"hyp-{key}": self.generate_walks(
                name=f"hyp-{key}", mixture=value, amount_walks=self.amount_walker
            ) for key, value in MIXTURES.items()}
            if self.isMixedtrails:
                curr_mixtrues = [(k, v, MIXTURES[k]) for k, v in self.mixture.items()]
                current_walks = [self.generate_walks(
                    name=f"data-{key}", mixture=value, amount_walks=int(self.amount_walker*procent)
                ) for key, procent ,value in curr_mixtrues]
                self.data_walks = {f"data-mixed": [item for sublist in current_walks for item in sublist]}
            else:
                self.data_walks = {f"data-{key}": self.generate_walks(
                    name=f"data-{key}", mixture=value, amount_walks=self.amount_walker
                ) for key, value in MIXTURES.items()}
        if self.verbose:
            print("\n\tWalks generated\n")

    def generate_walks(self, name: str, mixture: dict, amount_walks: int = 1):
        walks = []
        for _ in tqdm(
            range(amount_walks),
            desc=f"Generating {amount_walks} walks for mixture {name} ({mixture}) ...",
        ):
            for node in self.graph.nodes():
                walks.append(self.generate_walk(node, preference=mixture))
        return walks

    def generate_walk(self, start_node: int, preference: dict):
        changing_preference = True if isinstance(list(preference.keys())[0], float) else False
        if not changing_preference:
            if sum(preference.values()) != 1:
                return self.generate_higher_order_walk(start_node, preference)
        walk = [start_node]
        for _ in range(self.walk_length - 1):
            if changing_preference:
                current_relative_position = len(walk) / self.walk_length
                for curr_position, preference_dict in preference.items():
                    if current_relative_position < curr_position:
                        step_preference = np.random.choice(
                            list(preference_dict.keys()), p=list(preference_dict.values())
                        )
                        break
            else:
                step_preference = np.random.choice(list(preference.keys()), p=list(preference.values()))
            walk.append(self.generate_next_node(walk[-1], step_preference))
        return walk
    
    def generate_higher_order_walk(self, start_node, preference: dict):
        walk = [start_node]
        behavior = "even" if start_node % 2 == 0 else "odd"
        for _ in range(self.walk_length - 1):
            behavior_length = preference[behavior]
            if len(walk) >= behavior_length and all(node % 2 == 0 if behavior == "even" else node % 2 == 1 for node in walk[-behavior_length:]):
                # change behavior
                behavior = "even" if behavior == "odd" else "odd"
            walk.append(self.generate_next_node(walk[-1], behavior))
        return walk

    def generate_next_node(self, node, step_preference: str):
        neighbours = list(self.graph.neighbors(node))
        if step_preference == "even":
            neighbours = [n for n in neighbours if n % 2 == 0]
        elif step_preference == "odd":
            neighbours = [n for n in neighbours if n % 2 == 1]
        else:
            # let walker randomly teleport
            return random.choice(list(self.graph.nodes()))
        if len(neighbours) == 0:
            neighbours = list(self.graph.neighbors(node))
        return random.choice(neighbours)

    def build_subtrails_walks(self, walks_per_category: dict):
        used_categories = set(walks_per_category.values())
        all_walks = []
        for key, val in walks_per_category.items():
            mixture = MIXTURES[key]
            walks = self.generate_walks(
                name=key, mixture=mixture, amount_walks=self.amount_walker
            )
            for idx, walkie in enumerate(walks):
                all_walks.append(self.annotate_walk(idx, val, walkie, used_categories))
        return all_walks

    def annotate_walk(self, walk_idx: int, category: int, walk: list, used_categories: set):
        ret_walk = []
        for node in walk:
            category_dict = {}
            for i, cat in enumerate(self.categories):
                if category == i:
                    category_dict[cat] = 1
                elif i in used_categories:
                    category_dict[cat] = 0
                else:
                    category_dict[cat] = np.random.randint(2)
            ret_walk.append([
                walk_idx,
                node,
                node,
                category_dict,
            ])
        return ret_walk



if __name__ == "__main__":
    dataset = SyntheticDataset(
        args={
            "size": 100,
            "graph_type": "barabasi",
            "connectivity": 10,
            "amount_walker": 5,
            "walk_length": 30,
            "categories": ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6"],
            "walks_per_category": {'even': 0, 'odd': 1, 'first-even': 2},
            "verbose": 1,
        }
    )
    data_walks, hypothesis_walks = dataset.get_walks()
    print("Hypothesis walks")
    for key, val in hypothesis_walks.items():
        print(key, val[:2])
    print("Data walks")
    for key, val in data_walks.items():
        print(key, val[:2])
