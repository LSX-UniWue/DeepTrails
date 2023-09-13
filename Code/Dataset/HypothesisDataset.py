import json

import networkx as nx


class HypothesisDataset:
    def __init__(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)

        if "hypothesis_walks" in data:
            self.hypothesis_walks = data["hypothesis_walks"] 
        if "data_walks" in data:
            self.data_walks = data["data_walks"]
            self.isSubtrails = False
        if "annotated_walks" in data:
            self.isSubtrails = True
            self.annotated_walks = data["annotated_walks"]
        if data["graph"] is not None:
            self.graph = nx.from_dict_of_dicts(data["graph"])
        self.args = data["args"]

    def to_json(self):
        if self.isSubtrails:
            return json.dumps(
                {
                    "args": self.args,
                    "graph": nx.to_dict_of_dicts(self.graph) if self.graph is not None else None,
                    "annotated_walks": self.annotated_walks,
                },
            )
        # else ...
        return json.dumps(
            {
                "args": self.args,
                "graph": nx.to_dict_of_dicts(self.graph) if self.graph is not None else None,
                "hypothesis_walks": self.hypothesis_walks,
                "data_walks": self.data_walks,
            },
        )

    def to_file(self, path: str):
        with open(path, "w") as f:
            return f.write(self.to_json())

    def get_walks(self):
        return self.data_walks, self.hypothesis_walks
