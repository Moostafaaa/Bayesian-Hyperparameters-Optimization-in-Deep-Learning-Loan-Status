"""
objective.py
------------
The Optuna objective function — this is the Likelihood in the
Bayesian framework: P(AUC | θ).

For each hyperparameter configuration θ suggested by the TPE
surrogate, this function runs a full train/evaluate cycle and
returns the validation AUC as the observed evidence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.model import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_objective(X: np.ndarray, y: np.ndarray, input_dim: int):
    """
    Factory that captures the dataset in a closure and returns
    the objective function ready for study.optimize().

    Args:
        X:         Scaled training features (numpy array)
        y:         Binary labels (numpy array)
        input_dim: Number of input features

    Returns:
        objective(trial) → float (validation AUC)
    """

    def objective(trial: optuna.Trial) -> float:
        # ── Suggest training hyperparameters ──────────────────────────────
        # These suggest_* calls are the PRIOR: before any trial, each
        # parameter is drawn from its declared search space distribution.
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [256, 512, 1024])
        epochs       = trial.suggest_int("epochs", 5, 20)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        # ── Stratified validation split ───────────────────────────────────
        # StratifiedKFold preserves class ratio in both train and val folds,
        # critical because the loan default dataset is class-imbalanced.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(X, y))

        X_tr  = torch.tensor(X[train_idx], dtype=torch.float32)
        y_tr  = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
        X_val = torch.tensor(X[val_idx],   dtype=torch.float32)
        y_val = y[val_idx]

        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=batch_size, shuffle=True
        )

        model     = create_model(trial, input_dim).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()   # sigmoid + BCE fused, numerically stable

        # ── Training loop ─────────────────────────────────────────────────
        model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                # Gradient clipping guards against exploding gradients
                # on skewed financial features
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # ── LIKELIHOOD evaluation ─────────────────────────────────────
            # At the end of each epoch, compute AUC on the validation fold.
            # This is the evidence signal P(AUC | θ) fed back to the
            # Bayesian surrogate to update the posterior.
            model.eval()
            with torch.no_grad():
                logits = model(X_val.to(DEVICE))
                preds  = torch.sigmoid(logits).cpu().numpy().ravel()

            auc = roc_auc_score(y_val, preds)

            # Report intermediate value — enables the MedianPruner to
            # kill unpromising trials before they complete all epochs.
            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return auc   # Final LIKELIHOOD value for this configuration θ

    return objective
