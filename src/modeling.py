from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


def fit_baseline_lr(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_binary(model, X_val, y_val) -> dict[str, float]:
    p = model.predict_proba(X_val)[:, 1]
    return {
        "logloss": float(log_loss(y_val, p)),
        "roc_auc": float(roc_auc_score(y_val, p)),
        "mean_proba": float(np.mean(p)),
    }
