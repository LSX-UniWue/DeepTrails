import pickle
import uuid
from pathlib import Path

from Code.Baselines.baselines import *
from Code.Dataset.HypothesisDataset import HypothesisDataset
from Code.Dataset.ReviewDataset import ReviewsDataset, RFAMZWalkDataset
from Code.Dataset.SyntheticDataset import RFWalkDataset, SyntheticDataset


def main(
    save_path: Path,
    model_str: str,
    dataset_to_train: str,
):
    # Setup dataset
    if "amz_real_data" in str(save_path):
        dataset = ReviewsDataset(Path("data", "amz_real_data", "dataset.jsonl"))

    elif Path.exists(Path("data", save_path, "dataset.jsonl")):
        print("Loading dataset")
        dataset = HypothesisDataset(Path("data", save_path, "dataset.jsonl"))
    else:
        raise ValueError("No dataset found")

    walks, _ = dataset.get_walks()

    if dataset_to_train is not None and dataset_to_train in walks:
        walks = walks[dataset_to_train]
    else:
        print(ValueError(f"Dataset {dataset_to_train} not found"))

    if "amz_real_data" in str(save_path):
        train_dataset = RFAMZWalkDataset(walks, walk_type=dataset_to_train, args=dataset.args)
    else:
        train_dataset = RFWalkDataset(walks)

    if model_str == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=5,
            max_depth=20,
            criterion="log_loss",
            max_features=None,
            n_jobs=8,
        )
    else:  # cry
        raise ValueError("Model not implemented")
    try:
        model.feature_ordering = train_dataset.feature_ordering
    except AttributeError:
        pass

    # create a trainer and train the model
    model.fit(
        X=train_dataset.inputs,
        y=train_dataset.targets,
    )
    pickle.dump(model, open(Path("data", save_path, f"randoLM_{dataset_to_train}.pkl"), "wb"))


run_id = str(uuid.uuid4()).split("-")[0]

if __name__ == "__main__":
    COUNT_WORKER = 4
    MODEL_STR = "random_forest"

    for config in [
        {
            "save_path": "amz_real_data",
            "dataset_to_train": "fkt",
        },
        {
            "save_path": "amz_real_data",
            "dataset_to_train": "kat",
        },
    ] + [
        # {"save_path": "all-data", "dataset_to_train": f"data-{split}"}
        # for split in {
        #    "even",
        #    "odd",
        #    "first-even",
        #    "first-odd",
        #    "rand",
        #    "tele",
        #    "even-biased",
        #    "odd-biased",
        #    "first-even-biased",
        #    "first-odd-biased",
        #    "rand-biased",
        #    "tele-biased",
        # }
    ]:
        print(config)
        main(
            model_str=MODEL_STR,
            **config,
        )
