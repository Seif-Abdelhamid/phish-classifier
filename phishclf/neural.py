"""A small fastText-style neural classifier in PyTorch, used as a comparison model.

Each email becomes a bag of hashed word unigrams and bigrams. An ``EmbeddingBag`` learns a
vector for every hash bucket and averages the vectors of an email's tokens; one linear layer
turns that average into a phishing logit. Unlike TF-IDF, the model learns its own token
representation, so it is a genuinely different approach to the same task.

PyTorch is an optional dependency (``pip install -e .[torch]``); the rest of the package does
not import this module.
"""

from __future__ import annotations

import itertools
import re
import zlib

import numpy as np
import torch
from torch import nn

TOKEN_RE = re.compile(r"[a-z0-9$@'.]+|[!?]")
MAX_TOKENS = 2000


def token_ids(text: str, buckets: int) -> list[int]:
    """Hashed unigram + bigram ids. crc32 is stable across runs, unlike Python's ``hash``."""
    words = TOKEN_RE.findall(text.lower())[:MAX_TOKENS]
    grams = words + [f"{a} {b}" for a, b in itertools.pairwise(words)]
    ids = [zlib.crc32(g.encode()) % buckets for g in grams]
    return ids or [0]


class FastText(nn.Module):
    def __init__(self, buckets: int, dim: int) -> None:
        super().__init__()
        self.emb = nn.EmbeddingBag(buckets, dim, mode="mean")
        self.out = nn.Linear(dim, 1)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.out(self.emb(ids, offsets)).squeeze(1)


def _batch(encoded: list[list[int]], idx) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = [encoded[i] for i in idx]
    offsets = torch.tensor(np.cumsum([0] + [len(c) for c in chunks[:-1]]), dtype=torch.long)
    ids = torch.tensor([t for c in chunks for t in c], dtype=torch.long)
    return ids, offsets


class FastTextClassifier:
    """Fit / predict_proba wrapper with early stopping on validation loss."""

    def __init__(
        self,
        buckets: int = 1 << 18,
        dim: int = 64,
        epochs: int = 10,
        patience: int = 2,
        batch_size: int = 64,
        lr: float = 5e-3,
        seed: int = 0,
        threads: int = 4,
    ) -> None:
        self.buckets, self.dim, self.epochs, self.patience = buckets, dim, epochs, patience
        self.batch_size, self.lr, self.seed, self.threads = batch_size, lr, seed, threads
        self.model: FastText | None = None
        self.epochs_run = 0

    def _encode(self, texts) -> list[list[int]]:
        return [token_ids(t, self.buckets) for t in texts]

    def _logits(self, encoded: list[list[int]]) -> torch.Tensor:
        assert self.model is not None  # noqa: S101  # nosec B101
        self.model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(encoded), 512):
                out.append(self.model(*_batch(encoded, range(start, min(start + 512, len(encoded))))))
        return torch.cat(out)

    def fit(self, texts, labels, val_texts, val_labels) -> FastTextClassifier:
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        x, y = self._encode(texts), torch.tensor(labels, dtype=torch.float32)
        xv, yv = self._encode(val_texts), torch.tensor(val_labels, dtype=torch.float32)
        pos = float(y.sum())
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / max(pos, 1.0)))
        self.model = FastText(self.buckets, self.dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(self.epochs):
            self.model.train()
            order = rng.permutation(len(x))
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model(*_batch(x, idx)), y[idx])
                loss.backward()
                opt.step()
            val_loss = float(loss_fn(self._logits(xv), yv))
            self.epochs_run = epoch + 1
            if val_loss < best - 1e-4:
                best, bad = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        p = torch.sigmoid(self._logits(self._encode(texts))).numpy()
        return np.column_stack([1 - p, p])
