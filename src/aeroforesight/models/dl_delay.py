"""Deep-learning flight-delay predictor (DL component).

A feed-forward neural net (PyTorch) predicts the probability that a flight
arrives more than `delay_threshold_min` late, from network/weather/congestion
features. If PyTorch is unavailable the module transparently falls back to a
scikit-learn ``MLPClassifier`` so the platform still runs end-to-end.

The public surface (`train`, `predict_proba`, `save`, `load`) is identical
regardless of backend, so the serving layer never needs to care.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:  # DL backend
    import torch
    import torch.nn as nn

    _HAVE_TORCH = True
except Exception:  # pragma: no cover - exercised only when torch is absent
    _HAVE_TORCH = False


# --------------------------------------------------------------------------- #
# PyTorch network                                                             #
# --------------------------------------------------------------------------- #
if _HAVE_TORCH:

    class _DelayNet(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):  # noqa: D401
            return self.net(x).squeeze(-1)


class DelayModel:
    """Backend-agnostic delay classifier."""

    def __init__(self, hidden_dim: int = 64, epochs: int = 8, batch_size: int = 256, lr: float = 1e-3):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.backend = "torch" if _HAVE_TORCH else "sklearn"
        self._model = None
        self.metrics: dict[str, float] = {}

    # -- training ----------------------------------------------------------- #
    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_tr = self.scaler.fit_transform(X_tr).astype("float32")
        X_val = self.scaler.transform(X_val).astype("float32")

        if self.backend == "torch":
            self._train_torch(X_tr, y_tr)
        else:
            self._train_sklearn(X_tr, y_tr)

        auc = float(roc_auc_score(y_val, self._raw_proba(X_val)))
        self.metrics = {"val_auc": round(auc, 4), "n_train": int(len(X_tr)), "backend": self.backend}
        return self.metrics

    def _train_torch(self, X, y):  # pragma: no cover - depends on torch
        self._model = _DelayNet(X.shape[1], self.hidden_dim)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        n = len(Xt)
        self._model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                logits = self._model(Xt[idx])
                loss = loss_fn(logits, yt[idx])
                loss.backward()
                opt.step()

    def _train_sklearn(self, X, y):
        from sklearn.neural_network import MLPClassifier

        self._model = MLPClassifier(
            hidden_layer_sizes=(self.hidden_dim, self.hidden_dim // 2),
            max_iter=200,
            early_stopping=True,
            random_state=42,
        )
        self._model.fit(X, y)

    # -- inference ---------------------------------------------------------- #
    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        if self.backend == "torch":  # pragma: no cover
            self._model.eval()
            with torch.no_grad():
                logits = self._model(torch.from_numpy(np.asarray(X, dtype="float32")))
                return torch.sigmoid(logits).numpy()
        return self._model.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(np.asarray(X, dtype="float32")).astype("float32")
        return self._raw_proba(X)

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        payload = {
            "backend": self.backend,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "hidden_dim": self.hidden_dim,
        }
        if self.backend == "torch":  # pragma: no cover
            torch.save(self._model.state_dict(), path.with_suffix(".pt"))
            payload["in_dim"] = self._model.net[0].in_features
        else:
            payload["sklearn_model"] = self._model
        joblib.dump(payload, path.with_suffix(".joblib"))
        (path.with_suffix(".metrics.json")).write_text(json.dumps(self.metrics, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DelayModel":
        import joblib

        path = Path(path)
        payload = joblib.load(path.with_suffix(".joblib"))
        m = cls(hidden_dim=payload["hidden_dim"])
        m.backend = payload["backend"]
        m.scaler = payload["scaler"]
        m.metrics = payload["metrics"]
        if m.backend == "torch":  # pragma: no cover
            m._model = _DelayNet(payload["in_dim"], m.hidden_dim)
            m._model.load_state_dict(torch.load(path.with_suffix(".pt")))
        else:
            m._model = payload["sklearn_model"]
        return m
