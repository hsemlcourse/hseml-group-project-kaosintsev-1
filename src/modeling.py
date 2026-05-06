from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from src.config import RANDOM_STATE


def neg_log_loss_scorer(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return -log_loss(y, p)


neg_log_loss = make_scorer(neg_log_loss_scorer, greater_is_better=True, needs_proba=True)


def evaluate_binary(model, X_val, y_val):
    p = model.predict_proba(X_val)[:, 1]
    return {
        "logloss": float(log_loss(y_val, p)),
        "roc_auc": float(roc_auc_score(y_val, p)),
        "mean_proba": float(np.mean(p)),
    }


def fit_baseline_lr(X_train, y_train, random_state=RANDOM_STATE):
    m = LogisticRegression(max_iter=1000, random_state=random_state)
    m.fit(X_train, y_train)
    return m


def train_val_holdout(X, y, val_size=0.15, test_size=0.15, random_state=RANDOM_STATE):
    tmp_size = val_size + test_size
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=tmp_size, random_state=random_state, stratify=y
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=random_state, stratify=y_tmp
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def cv_lgbm_mean_logloss(X, y, params, n_splits=5, random_state=RANDOM_STATE):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_i, val_i in kf.split(X, y):
        X_tr = X.iloc[train_i]
        X_va = X.iloc[val_i]
        y_tr = y.iloc[train_i]
        y_va = y.iloc[val_i]
        clf = LGBMClassifier(
            **params,
            random_state=random_state,
            verbosity=-1,
            force_row_wise=True,
        )
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_va)[:, 1]
        scores.append(log_loss(y_va, p))
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = var**0.5
    return mean, std


def random_search_lgbm(X, y, n_iter=12, cv=5, random_state=RANDOM_STATE):
    base = LGBMClassifier(random_state=random_state, verbosity=-1, force_row_wise=True)
    grid = {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.02, 0.05, 0.08, 0.1],
        "num_leaves": [31, 63, 127],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0, 5.0],
    }
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        base,
        param_distributions=grid,
        n_iter=n_iter,
        scoring=neg_log_loss,
        cv=kf,
        random_state=random_state,
        n_jobs=-1,
        refit=False,
        verbose=0,
    )
    search.fit(X, y)
    return search


def lgbm_from_search(search, random_state=RANDOM_STATE):
    d = dict(search.best_params_)
    d["random_state"] = random_state
    d["verbosity"] = -1
    d["force_row_wise"] = True
    return LGBMClassifier(**d)


def write_submission(kaggle_id, proba, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"ID": kaggle_id.values, "PredictedProb": proba})
    sub.to_csv(path, index=False)
