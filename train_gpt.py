import uuid
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from Code.Dataset.HypothesisDataset import HypothesisDataset
from Code.Dataset.ReviewDataset import AMZWalkDataset, ReviewsDataset
from Code.Dataset.SyntheticDataset import SyntheticDataset, WalkDataset
from Code.Models.LightningGPT import GPT, GPTConfig


def main(
    size: int,
    walk_length: int,
    train_split: float,
    save_path: Path,
    batch_size: int,
    run_id: str,
    dataset_to_train: str = None,
):
    # Setup dataset
    if "amz_real_data" in str(save_path):
        dataset = ReviewsDataset(Path("data", "amz_real_data", "dataset.jsonl"))
    elif "subtrails" in dataset_to_train:
        if Path.exists(Path("data", str(save_path), "dataset-subtrails.jsonl")):
            print(f"Loading subtrails dataset from data/{str(save_path)}/dataset-subtrails.jsonl")
            dataset = ReviewsDataset(Path("data", str(save_path), "dataset-subtrails.jsonl"))
        else:
            raise ValueError(f"No dataset found at {save_path}/dataset-subtrails.jsonl")
    elif "mixed" in dataset_to_train:
        if Path.exists(Path("data", str(save_path), "dataset-mixedtrail.jsonl")):
            print(f"Loading mixedtrails dataset from data/{str(save_path)}/dataset-mixedtrail.jsonl")
            dataset = HypothesisDataset(Path("data", str(save_path), "dataset-mixedtrail.jsonl"))
        else:
            raise ValueError(f"No dataset found at {save_path}/dataset-mixedtrail.jsonl")
    elif Path.exists(Path("data", str(save_path), "dataset.jsonl")):
        print(f"Loading hypothesis dataset from data/{str(save_path)}/dataset.jsonl")
        dataset = HypothesisDataset(Path("data", str(save_path), "dataset.jsonl"))
    else:
        raise ValueError(f"No dataset found (data/{str(save_path)}/dataset.jsonl))")

    walks, _ = dataset.get_walks()

    if not "amz_real_data" in str(save_path):
        if "subtrails" in dataset_to_train:
            curr_dataset = "_".join(dataset_to_train.split("-")[2:])
        else:
            curr_dataset = dataset_to_train
        if curr_dataset is not None and curr_dataset in walks:
            walks = walks[curr_dataset]
        else:
            raise ValueError(f"Dataset {curr_dataset} not found, available datasets: {list(walks.keys())}")

    if "amz_real_data" in str(save_path):
        train_dataset = AMZWalkDataset(
            walks,
            walk_type=dataset_to_train,
            args=dataset.args,
            max_length=walk_length,
        )

        train_loader = DataLoader(
            train_dataset,
            num_workers=COUNT_WORKER,
            batch_size=32,
            shuffle=True,
        )
        feature_embedding = 25
    elif "subtrails" in dataset_to_train:
        train_dataset = AMZWalkDataset(
            walks,
            walk_type=dataset_to_train,
            args=dataset.args
        )

        train_loader = DataLoader(
            train_dataset,
            num_workers=COUNT_WORKER,
            batch_size=32,
            shuffle=True,
        )   
        feature_embedding = 12
    else:
        train_walks = walks[: int(len(walks) * train_split)]
        train_dataset = WalkDataset(train_walks)
        train_loader = DataLoader(
            train_dataset,
            num_workers=COUNT_WORKER,
            batch_size=batch_size,
            shuffle=True,
        )
        val_walks = walks[int(len(walks) * train_split) :]
        val_dataset = WalkDataset(val_walks)
        val_loader = DataLoader(
            val_dataset,
            num_workers=COUNT_WORKER,
            batch_size=batch_size,
        )
        feature_embedding = 1

    config = GPTConfig(
        block_size=walk_length + 1,  # to be able to predict EOS in the end
        vocab_size=size + 2,  # 0 for BOS and 1 for EOS, all other tokens are thus shifted by 2
        n_layer=4,
        n_head=4,
        n_embd=16,
        feature_embd_dim=feature_embedding,
        bias=False,
    )
    model = GPT(config)

    # create a trainer and train the model
    trainer = L.Trainer(
        callbacks=[EarlyStopping(monitor="train_loss", mode="min", check_on_train_epoch_end=False, patience=5),
                   ModelCheckpoint(monitor="train_loss", 
                                   mode="min", 
                                   save_top_k=1, 
                                   dirpath=f"data/{save_path}", 
                                   filename=f"model_{dataset_to_train}_{run_id}")],
        max_epochs=30 if not "amz_real_data" in str(save_path) else 5,
        # val_check_interval=1,        
        check_val_every_n_epoch=1,
        # fast_dev_run=True,
    )
    trainer.fit(
        model,
        train_loader,
        val_loader if "amz_real_data" not in str(save_path) and "subtrails" not in dataset_to_train else None, 
    )
    trainer.save_checkpoint(Path("data", save_path, f"model_{dataset_to_train}_final_{run_id}.ckpt"))


def create_dataset(argv: str, 
                   save_path, size: int, 
                   connectivity: int, 
                   amount_walker: int, 
                   walk_length: int, 
                   mixedtrails_mixture: dict,
                   subtrails_categories: list, 
                   subtrails_walks_per_category: dict, 
                   verbose: int):
    if not Path.exists(save_path):
        Path.mkdir(save_path, parents=True)
    args = {
        "graph_type": "barabasi",
        "size": size,
        "connectivity": connectivity,
        "amount_walker": amount_walker,
        "walk_length": walk_length,
        "verbose": verbose,
    }

    if "subtrails" in argv:
        args['categories'] = subtrails_categories
        args['walks_per_category'] = subtrails_walks_per_category  # for this type of walk, the respective cat will always be set to 1  
        dataset_path = Path("data", save_path, "dataset-subtrails.jsonl")
    elif "mixedtrail" in argv:
        args['mixture'] = mixedtrails_mixture
        dataset_path = Path("data", save_path, "dataset-mixedtrail.jsonl")
    else:
        dataset_path = Path("data", save_path, "dataset.jsonl")
    dataset = SyntheticDataset(args=args)
    dataset.to_file(dataset_path)
    print("Dataset saved")


run_id = str(uuid.uuid4()).split("-")[0]

if __name__ == "__main__":
    import sys
    # Usage: python train_gpt.py 
    # args: 
    # - data to create dataset only, 
    # - data-subtrails to create subtrails dataset 
    # - <dataset_name> (e.g. "even") to run a model
    # - amz-fkt to run a model on the fkt dataset
    # - amz-kat to run a model on the kat dataset

    COUNT_WORKER = 4
    ARGV = sys.argv[1]
    # ARGV = "amz-kat"
    if "data" in ARGV:
        dataset_to_train = None
        walk_length = 20
        create_dataset_only = True
        save_path = Path("all-data")
    elif ARGV == "amz-fkt":
        dataset_to_train = "fkt"
        create_dataset_only = False
        save_path = Path("amz_real_data")
        walk_length = 200
        print(f"Training on Real World {dataset_to_train}")
    elif ARGV == "amz-kat":
        dataset_to_train = "kat"
        create_dataset_only = False
        save_path = Path("amz_real_data")
        walk_length = 200
        print(f"Training on Real World {dataset_to_train}")
    else:
        dataset_to_train = f"data-{ARGV}"
        walk_length = 20
        create_dataset_only = False
        save_path = Path("all-data")
        print(f"Training on {dataset_to_train}")

    if create_dataset_only:
        create_dataset(
            argv=ARGV,
            save_path=save_path,
            size=100,
            connectivity=10,
            amount_walker=1000,
            walk_length=walk_length,
            mixedtrails_mixture={"even": 0.33, "odd": 0.33, "first-even": 0.34},
            subtrails_categories=["cat1", "cat2", "cat3", "cat4", "cat5", "cat6"],
            subtrails_walks_per_category = {'even': 0, 'odd': 1, 'first-even': 2, 'first-odd': 3},
            verbose=1,
        )
    else:
        main(
            size=100,
            walk_length=walk_length,
            train_split=0.95,
            save_path=save_path,
            run_id=run_id,
            dataset_to_train=dataset_to_train,
            batch_size=512,
        )
