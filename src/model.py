"""
model.py
--------
Dynamic MLP factory driven by an Optuna trial object.

The architecture is itself a hyperparameter:
  - Number of layers is suggested by the Bayesian sampler
  - Width, BatchNorm, and dropout per layer are all tunable
  - Output is a single raw logit (no sigmoid) for BCEWithLogitsLoss
"""

import torch.nn as nn
import optuna


def create_model(trial: optuna.Trial, input_dim: int) -> nn.Sequential:
    """
    Build a dynamic MLP where the architecture is controlled by the
    Optuna trial's TPE surrogate. Each suggest_* call records a
    hyperparameter observation that updates the Bayesian posterior.

    Args:
        trial:     Optuna trial object (carries the TPE sampler state)
        input_dim: Number of input features (20 for the loan dataset)

    Returns:
        nn.Sequential MLP ending in a single linear output (raw logit).
    """
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers = []
    in_features = input_dim

    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 8, 128, step=8)
        layers.append(nn.Linear(in_features, out_features))

        # BatchNorm is itself a categorical hyperparameter —
        # the posterior will discover whether it helps for this dataset
        use_bn = trial.suggest_categorical(f"use_bn_l{i}", [True, False])
        if use_bn:
            layers.append(nn.BatchNorm1d(out_features))

        layers.append(nn.ReLU())

        p = trial.suggest_float(f"dropout_l{i}", 0.1, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features

    # Single logit output — no Sigmoid here.
    # BCEWithLogitsLoss fuses sigmoid + BCE in one numerically stable op.
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)
