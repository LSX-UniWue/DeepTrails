import json
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

from pytrails.hyptrails import MarkovChain
from Code.Dataset.SyntheticDataset import HypothesisDataset

class Eval:
    def __init__(self, args: dict):
        self.name = args["name"]
        self.k_factors = args["k_factors"]
        self.save_path = args["save_path"]
        self.hypothesis_to_plot = args["hypothesis_to_plot"]
        self.create_plot = args["create_plot"]
        self.verbose = args["verbose"]
        if args['create_evidences']:
            self.dataset = HypothesisDataset(Path("data", str(self.save_path), "dataset.jsonl"))
            data_walks, hypothesis_walks = self.dataset.get_walks()
            self.size = np.array(data_walks["data-two-even-two-odd"]).max() + 1
            if self.verbose:
                print("\n\tAccumulating walks\n")
            self.data = self.accumulate_walks(data_walks['data-two-even-two-odd'], "data", normalize=False)
            self.hypotheses = {key: self.accumulate_walks(walks, key) for key, walks in hypothesis_walks.items()}
            self.evidences = {}
            if self.verbose:
                print("\n\tCalculating evidences\n")
            self.evidences = {
                key: [MarkovChain.marginal_likelihood(self.data, hypothesis * ks) for ks in self.k_factors]
                for key, hypothesis in self.hypotheses.items()
            }
            with open(Path("data", self.save_path, f"{self.name}.json"), "w") as f:
                json.dump(self.evidences, f)
        else:
            with open(Path("data", self.save_path, f"{self.name}.json"), "r") as f:
                self.evidences = json.load(f)
        if self.create_plot:
            self.plot(
                evidences=self.evidences, k_values=self.k_factors, save_path=Path("data", "potential-paper-figures", f"image-{self.name}.pdf")
            )

    @staticmethod
    def save_normalize(matrix) -> csr_matrix:
        for row in matrix:
            if row.sum() != 0:
                row /= row.sum()
        return matrix

    def accumulate_walks(self, walks, key, normalize=True) -> csr_matrix:
        if self.verbose:
            print(f"\tAccumulating walks for {key}")
        matrix = csr_matrix((self.size, self.size))
        for walk in walks:
            for i in range(len(walk) - 1):
                matrix[walk[i], walk[i + 1]] += 1
        if normalize:
            matrix = self.save_normalize(matrix)
        return matrix

    def plot(self, evidences: dict, k_values: list, save_path: str) -> None:
        import matplotlib.pyplot as plt
        tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 18,
            "font.size": 18,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
        plt.rcParams.update(tex_fonts)
        plt.figure()
        for name, evidence in evidences.items():
            if name in self.hypothesis_to_plot:
                plt.plot(np.arange(len(evidence)), evidence, label=name.replace("hyp-", "").title())
        plt.xticks(np.arange(len(evidence)), k_values)
        plt.xlabel("k-factor")
        plt.ylabel("Evidence")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)

    def get_evidences(self):
        return self.evidences


if __name__ == "__main__":
    Eval(
        args={
            "k_factors": [0, 1, 3, 5, 10, 100, 1000],
            "save_path": "all-data",
            "name": "hyptrails",
            "create_evidences": False,
            "create_plot": True,
            "hypothesis_to_plot": ["hyp-two-even-two-odd", "hyp-even", "hyp-odd", "hyp-rand", "hyp-tele"],
            "verbose": 1,
        },
    )
