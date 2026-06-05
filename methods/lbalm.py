"""
LBALM: Log-Barrier Augmented Lagrangian Method
Proposed Method I — ICDM 2026

Paper: As Fair as Possible: Constrained Optimization of
       Fairness under Accuracy Bounds

Minimizes unfairness R(theta) subject to:
    g(theta) = L_val(theta) - (B + delta) <= 0

Objective:
    J = R(theta) + lambda * g + (rho/2) * g^2
        - (1/t) * log(max(-g, eps))
"""

import torch
import torch.nn as nn
import numpy as np


def p_rule_mp(probs, sensitive):
    """
    Differentiable mean-probability p%-rule proxy.
    Args:
        probs:     predicted probabilities (N,)
        sensitive: binary sensitive attribute (N,)
    Returns:
        p%-rule value in [0, 1]
    """
    mask0 = (sensitive == 0).float()
    mask1 = (sensitive == 1).float()
    mean0 = (probs * mask0).sum() / (mask0.sum() + 1e-8)
    mean1 = (probs * mask1).sum() / (mask1.sum() + 1e-8)
    return torch.min(mean0, mean1) / (torch.max(mean0, mean1) + 1e-8)


def train_lbalm(
    model,
    train_loader,
    val_loader,
    anchor_X,
    anchor_s,
    B,
    delta,
    rho=0.5,
    lambda_init=0.1,
    t=1.0,
    lambda_max=100.0,
    eps=1e-6,
    lr=5e-4,
    epochs=50,
    device="cpu",
):
    """
    Train a model using LBALM.

    Args:
        model:        PyTorch neural network
        train_loader: DataLoader for training data
        val_loader:   DataLoader for validation data
        anchor_X:     anchor set features (fixed, for fairness gradient)
        anchor_s:     anchor set sensitive attributes
        B:            baseline validation loss
        delta:        performance budget
        rho:          quadratic penalty coefficient
        lambda_init:  initial dual variable
        t:            fixed barrier parameter
        lambda_max:   dual variable upper bound
        eps:          numerical stability constant
        lr:           learning rate
        epochs:       number of training epochs
        device:       'cpu' or 'cuda'

    Returns:
        model:        trained model
        history:      dict with per-epoch metrics
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    criterion = nn.BCELoss()

    lam = torch.tensor(lambda_init, dtype=torch.float32)
    history = {"g_val": [], "p_rule": [], "loss": []}

    anchor_X = anchor_X.to(device)
    anchor_s = anchor_s.to(device)

    for epoch in range(epochs):
        model.train()

        for X_b, y_b, _ in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            # ── Fairness objective on anchor set ──────────────────
            probs_anchor = torch.sigmoid(model(anchor_X)).squeeze()
            R = 1.0 - p_rule_mp(probs_anchor, anchor_s)

            # ── Constraint on mini-batch ──────────────────────────
            probs_b = torch.sigmoid(model(X_b)).squeeze()
            loss_b = criterion(probs_b, y_b.float())
            g = loss_b - (B + delta)

            # ── LBALM objective ───────────────────────────────────
            barrier = -torch.log(torch.clamp(-g, min=eps)) / t
            J = R + lam.item() * g + (rho / 2) * g ** 2 + barrier

            J.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # ── Dual update on full validation set (once per epoch) ──
        model.eval()
        with torch.no_grad():
            val_losses, val_probs, val_s = [], [], []
            for X_v, y_v, s_v in val_loader:
                X_v, y_v = X_v.to(device), y_v.to(device)
                p_v = torch.sigmoid(model(X_v)).squeeze()
                val_losses.append(
                    criterion(p_v, y_v.float()).item()
                )
                val_probs.append(p_v.cpu())
                val_s.append(s_v)

            L_val = np.mean(val_losses)
            g_val = L_val - (B + delta)

            all_probs = torch.cat(val_probs)
            all_s = torch.cat(val_s)
            p_rule_val = p_rule_mp(
                all_probs, all_s.float()
            ).item()

        # Dual update
        lam = torch.clamp(
            lam + rho * g_val, min=0.0, max=lambda_max
        )

        history["g_val"].append(g_val)
        history["p_rule"].append(p_rule_val)
        history["loss"].append(L_val)

        print(
            f"Epoch {epoch+1:3d} | "
            f"g_val={g_val:+.4f} | "
            f"p%-rule={p_rule_val:.4f} | "
            f"lambda={lam.item():.4f}"
        )

    return model, history
