# @title minimal GPT implementation in PyTorch (optional)
""" super minimal decoder-only gpt """

import math

import lightning as L
import torch
import torch.nn as nn
from torch.nn import functional as F

from Code.Models.nanoGPT import GPT as nanoGPT
from Code.Models.nanoGPT import GPTConfig

torch.manual_seed(1337)


class GPT(nanoGPT, L.LightningModule):
    def configure_optimizers(self):
        return super().configure_optimizers(
            weight_decay=1e-1,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            device_type=str(self.device).split(":")[0],
        )

    def _teacher_force(self, batch: torch.Tensor, targets: torch.Tensor = None, split: str = None):
        assert split in ["train", "val", "test"]
        if type(batch) == list:
            features = batch[1]
            batch = batch[0]
        else:
            features = None

        if not targets:
            # right shift the targets
            targets = batch[:, 1:].contiguous()
            # don't feed in EOS
            batch = batch[:, :-1].contiguous()

        logits, loss = self(batch, features=features, targets=targets)
        # at which rank are the correct target tokens in the predicted logits for each position?
        likeliest_tokens_ranked = logits.argsort(2, descending=True)
        ranks = (targets.unsqueeze(-1) == likeliest_tokens_ranked).nonzero()[:, -1]

        self.log(f"{split}_loss", loss)
        self.log(f"{split}_acc", (ranks == 0).float().mean())
        self.log(f"{split}_mrr", (1 / (ranks + 1)).mean())
        self.log(f"{split}_hit10", (ranks < 10).float().mean())
        self.log(f"{split}_hit20", (ranks < 20).float().mean())
        self.log(f"{split}_ndcg", (1 / torch.log2(ranks + 2)).mean())
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx):
        return self._teacher_force(batch, split="train")

    def validation_step(self, batch, batch_idx):
        return self._teacher_force(batch, split="val")

    def test_step(self, batch, batch_idx):
        return self._teacher_force(batch, split="test")
