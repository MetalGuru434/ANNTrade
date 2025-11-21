"""
End-to-end forecasting baseline for lotto_zahlen.txt using multiple modeling
approaches (classical ML, probabilistic, and neural sequence models).

Usage:
    python lotto_pipeline.py

The script reads lotto_zahlen.txt, builds training/validation/test splits,
trains several models (Logistic Regression, Gradient Boosting, Markov chain
baseline, LSTM, GRU, 1D CNN, Transformer) and reports per-model macro F1 and
hit-rate@6 on the held-out test split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

RANDOM_SEED = 42
NUM_NUMBERS = 49
NUM_PICKS = 6

def load_draws(path: str) -> np.ndarray:
    """Load draws from a whitespace-separated file of six integers per line."""
    data = np.loadtxt(path, dtype=int)
    if data.shape[1] != NUM_PICKS:
        raise ValueError(f"Expected {NUM_PICKS} numbers per draw, got {data.shape[1]}")
    return data

def draws_to_binary(draws: np.ndarray) -> np.ndarray:
    binary = np.zeros((draws.shape[0], NUM_NUMBERS), dtype=np.float32)
    for i, row in enumerate(draws):
        binary[i, row - 1] = 1.0
    return binary

def build_sequences(binary_draws: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    for idx in range(lookback, len(binary_draws)):
        features.append(binary_draws[idx - lookback : idx])
        targets.append(binary_draws[idx])
    return np.stack(features), np.stack(targets)

@dataclass
class SplitData:
    train: Tuple[np.ndarray, np.ndarray]
    val: Tuple[np.ndarray, np.ndarray]
    test: Tuple[np.ndarray, np.ndarray]


def sequential_split(features: np.ndarray, targets: np.ndarray, val_ratio: float = 0.15, test_ratio: float = 0.15) -> SplitData:
    n_samples = len(features)
    test_size = int(n_samples * test_ratio)
    val_size = int(n_samples * val_ratio)
    train_end = n_samples - test_size - val_size
    val_end = n_samples - test_size
    return SplitData(
        train=(features[:train_end], targets[:train_end]),
        val=(features[train_end:val_end], targets[train_end:val_end]),
        test=(features[val_end:], targets[val_end:]),
    )


def top_k_predictions(probabilities: np.ndarray, k: int = NUM_PICKS) -> np.ndarray:
    topk_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]
    preds = np.zeros_like(probabilities)
    for row, cols in enumerate(topk_indices):
        preds[row, cols] = 1
    return preds


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    hits = (np.logical_and(y_true == 1, y_pred == 1).sum(axis=1))
    hit_rate = (hits > 0).mean()
    return f1, hit_rate


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class GRUForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class CNN1DForecaster(nn.Module):
    def __init__(self, input_size: int, lookback: int, channels: int = 32):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=lookback, out_channels=channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class TransformerForecaster(nn.Module):
    def __init__(self, input_size: int, lookback: int, emb_dim: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(emb_dim, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.input_proj(x)
        encoded = self.encoder(emb)
        return self.head(encoded[:, -1, :])


def train_torch_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10, lr: float = 1e-3, device: str = "cpu") -> nn.Module:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_torch_model(model: nn.Module, loader: DataLoader, device: str = "cpu") -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(preds)


def markov_baseline(train_targets: np.ndarray, lookback_features: np.ndarray) -> np.ndarray:
    last_draws = lookback_features[:, -1, :]
    counts = np.zeros(NUM_NUMBERS, dtype=np.float32)
    for prev, nxt in zip(last_draws, train_targets):
        counts += nxt * prev
    global_counts = train_targets.sum(axis=0)
    probabilities = (counts + global_counts) / (counts.sum() + global_counts.sum())
    return probabilities


def run_classical_models(train: Tuple[np.ndarray, np.ndarray], evaluate_on: Tuple[np.ndarray, np.ndarray]) -> dict:
    x_train, y_train = train
    x_eval, y_eval = evaluate_on
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_eval_flat = x_eval.reshape(len(x_eval), -1)

    scaler = StandardScaler()

    logreg = MultiOutputClassifier(LogisticRegression(max_iter=500, n_jobs=-1))
    gb = MultiOutputClassifier(GradientBoostingClassifier())

    models = {
        "Logistic Regression": Pipeline([("scale", scaler), ("clf", logreg)]),
        "Gradient Boosting": Pipeline([("scale", scaler), ("clf", gb)]),
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train_flat, y_train)
        proba = model.predict_proba(x_eval_flat)
        stacked = np.stack([p[:, 1] for p in proba], axis=1)
        preds = top_k_predictions(stacked)
        f1, hit = evaluate_predictions(y_eval, preds)
        results[name] = (f1, hit)
    return results


def run_neural_models(train: Tuple[np.ndarray, np.ndarray], evaluate_on: Tuple[np.ndarray, np.ndarray], lookback: int, device: str = "cpu") -> dict:
    x_train, y_train = train
    x_eval, y_eval = evaluate_on
    train_ds = SequenceDataset(x_train, y_train)
    eval_ds = SequenceDataset(x_eval, y_eval)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=64)

    models = {
        "LSTM": LSTMForecaster(NUM_NUMBERS),
        "GRU": GRUForecaster(NUM_NUMBERS),
        "1D CNN": CNN1DForecaster(NUM_NUMBERS, lookback=lookback),
        "Transformer": TransformerForecaster(NUM_NUMBERS, lookback=lookback),
    }

    results = {}
    for name, model in models.items():
        trained = train_torch_model(model, train_loader, eval_loader, device=device)
        proba = predict_torch_model(trained, eval_loader, device=device)
        preds = top_k_predictions(proba)
        f1, hit = evaluate_predictions(y_eval, preds)
        results[name] = (f1, hit)
    return results


def summarize_results(results: dict) -> None:
    print("Model\tMacro F1\tHit@6")
    for name, (f1, hit) in results.items():
        print(f"{name}\t{f1:.3f}\t{hit:.3f}")


def main(args: argparse.Namespace) -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    draws = load_draws(args.data_path)
    binary = draws_to_binary(draws)
    features, targets = build_sequences(binary, args.lookback)
    split = sequential_split(features, targets)

    classical_results = run_classical_models(split.train, split.test)

    markov_probs = markov_baseline(split.train[1], split.train[0])
    markov_pred = top_k_predictions(np.tile(markov_probs, (len(split.test[1]), 1)))
    markov_scores = evaluate_predictions(split.test[1], markov_pred)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    neural_results = run_neural_models(split.train, split.test, args.lookback, device=device)

    all_results = {**classical_results, "Markov": markov_scores, **neural_results}
    summarize_results(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast lottery draws using multiple models.")
    parser.add_argument("--data-path", default="lotto_zahlen.txt", help="Path to lotto data file")
    parser.add_argument("--lookback", type=int, default=5, help="Number of past draws for forecasting")
    main(parser.parse_args())
