"""
neural_models.py

Architecture comparison, in the spirit of Ex.2 and Ex.4: don't just
train one model, compare a small progression of them on the same
train/test split used by baseline_models.py (Ridge/RF from the ML
project).

Planned comparison:
  1. baseline_models.py - Ridge / RF on hand-picked features
     (frequency, concreteness, length) - already built, ML project
  2. dense_nn_on_features() - small feed-forward NN, same hand-picked
     features as (1) - isolates "did switching to a NN help, holding
     features constant?" (Stretch)
  3. dense_nn_on_embeddings() - same NN shape, but fed pretrained
     transformer embeddings from embeddings.py instead - isolates
     "did switching features to embeddings help, holding architecture
     constant?" (Core)

Keep the SAME train/test split (from data_prep.split_raw, ML project)
across all three, or the comparison isn't valid - this was exactly
the leakage-adjacent lesson from the ML project resubmission.
"""

import torch
import torch.nn as nn


class DenseNet(nn.Module):
    """
    Small feed-forward network. Same architecture class used for
    both the hand-features and embeddings variants below - only the
    input dimension changes, so the comparison isolates the feature
    representation, not the model capacity.

    TODO: decide hidden layer sizes, dropout, activation - given the
    small dataset and high-dimensional embeddings (768-dim from
    MacBERT), keep this small and regularized to avoid overfitting.
    """

    def __init__(self, input_dim, hidden_dim=32, dropout=0.2):
        super().__init__()
        # TODO: implement layers
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def train_dense_nn(X_train, y_train, X_val, y_val, **kwargs):
    """
    Training loop: TODO implement with an optimizer from Ex.2
    (Adam/AdamW are reasonable defaults), track train/val loss
    per epoch, and plot learning curves.
    """
    raise NotImplementedError


def evaluate_nn(model, X_test, y_test):
    """
    Return same metrics as baseline_models.py (R^2, etc.) so results
    are directly comparable in one results table.
    """
    raise NotImplementedError
