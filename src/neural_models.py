"""
neural_models.py

Architecture comparison, in the spirit of Ex.2 and Ex.4: don't just
train one model, compare a small progression of them on the same
train/test split used by baseline_models.py (Ridge/RF from the ML
project).

Comparison this supports:
  1. baseline_models.py - Ridge / RF on hand-picked features
     (frequency, concreteness, length) - Core
  2. dense NN on the same hand-picked features - isolates
     "did switching to a NN help, holding features constant?" (Stretch)
  3. dense NN on pretrained transformer embeddings (embeddings.py) -
     isolates "did switching features to embeddings help, holding
     architecture constant?" (Core)

The SAME train/test split (idx_*_train / idx_*_test from
baseline_models.split_raw) is reused across all three - this is the
leakage-adjacent lesson from the ML project resubmission, applied
here to keep the architecture comparison valid.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def make_train_val_split(X_train, y_train, val_size=0.2, random_state=42):
    """
    Carve a validation set out of an existing training set, for
    early stopping during NN training. Call this on X_train/y_train
    only - never on X_test/y_test, which must stay untouched until
    final evaluation.
    """
    return train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)


class DenseNet(nn.Module):
    """
    Small feed-forward network. Same architecture class used for
    both the hand-features (3-dim input) and embeddings (768-dim
    input) variants - only input_dim changes, so the comparison
    isolates the feature representation, not the model capacity.

    Kept deliberately small and regularized (dropout) given the
    small dataset size, especially for the high-dimensional
    embeddings case where overfitting risk is real.
    """

    def __init__(self, input_dim, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_dense_nn(X_train, y_train, X_val, y_val, hidden_dim=32,
                    dropout=0.2, lr=1e-3, weight_decay=1e-4,
                    epochs=300, patience=20, batch_size=32, verbose=True,
                    seed=42):
    """
    Train a DenseNet with early stopping on validation loss.

    X_train/X_val should already be scaled (StandardScaler fit on
    the training portion only) before calling this - scaling is
    handled at the notebook level, same discipline as the Pipeline
    approach in baseline_models.py, not inside this function.

    `seed` fixes weight initialization and batch shuffling, so
    repeated runs on the same data give identical results - matches
    the random_state=42 convention already used everywhere else in
    the project (split_raw, KFold, make_train_val_split).

    Returns (model, history) where history has per-epoch train/val
    losses, for learning-curve plotting.
    """
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train_t = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(device)
    y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32).to(device)
    X_val_t = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(device)
    y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.float32).to(device)

    input_dim = X_train_t.shape[1]
    model = DenseNet(input_dim, hidden_dim, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    n = X_train_t.shape[0]
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1} "
                          f"(best val loss: {best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_nn(model, X_test, y_test):
    """
    Return R2/MAE/RMSE - same metrics reported for Ridge in
    baseline_models.py, so results are directly comparable in one
    results table (Section 5).
    """
    device = next(model.parameters()).device
    model.eval()
    X_test_t = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()

    return {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'y_pred': y_pred,
    }
