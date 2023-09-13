import torch

from dataclasses import dataclass
import torch.nn as nn
import lightning as L
from torch.nn import functional as F

torch.manual_seed(1337)


@dataclass
class MLPConfig:
    block_size: int = 10
    vocab_size: int = 100
    n_layer: int = 2
    n_embd: int = 16
    bias: bool = True


class MLP(L.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.mlp = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(config.vocab_size, config.n_embd),
                mlp=nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_layer)]),
                projection=nn.Linear(config.n_embd * (config.block_size - 1), config.vocab_size, bias=config.bias),
                nonlin=nn.GELU(),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp.embedding(x)
        for layer in self.mlp.mlp:
            x = layer(x)
            x = self.mlp.nonlin(x)
        x = self.mlp.projection(x.view(x.shape[0], -1))
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        logits = self(batch[:, :-1])
        loss = F.cross_entropy(logits, batch[:, -1:].reshape(-1))
        ranks = (logits.argsort(dim=-1, descending=True) == batch[:, -1:].reshape(-1, 1)).nonzero()[:, -1].float()
        self.log("train_loss", loss)
        self.log("train_acc", (ranks == 0).float().mean())
        self.log("train_mrr", (1 / (ranks + 1)).mean())
        self.log("train_hit10", (ranks < 10).float().mean())
        self.log("train_hit20", (ranks < 20).float().mean())
        self.log("train_ndcg", (1 / torch.log2(ranks + 2)).mean())
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch[:, :-1])
        loss = F.cross_entropy(logits, batch[:, -1:].reshape(-1))
        ranks = (logits.argsort(dim=-1, descending=True) == batch[:, -1:].reshape(-1, 1)).nonzero()[:, -1].float()
        self.log("val_loss", loss)
        self.log("val_acc", (ranks == 0).float().mean())
        self.log("val_mrr", (1 / (ranks + 1)).mean())
        self.log("val_hit10", (ranks < 10).float().mean())
        self.log("val_hit20", (ranks < 20).float().mean())
        self.log("val_ndcg", (1 / torch.log2(ranks + 2)).mean())
        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch[:, :-1])
        loss = F.cross_entropy(logits, batch[:, -1:].reshape(-1))
        ranks = (logits.argsort(dim=-1, descending=True) == batch[:, -1:].reshape(-1, 1)).nonzero()[:, -1].float()
        self.log("test_loss", loss)
        self.log("test_acc", (ranks == 0).float().mean())
        self.log("test_mrr", (1 / (ranks + 1)).mean())
        self.log("test_hit10", (ranks < 10).float().mean())
        self.log("test_hit20", (ranks < 20).float().mean())
        self.log("test_ndcg", (1 / torch.log2(ranks + 2)).mean())
        return loss, ranks
