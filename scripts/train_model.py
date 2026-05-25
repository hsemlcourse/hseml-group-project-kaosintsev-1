#!/usr/bin/env python3
"""Обучение финальной модели и сохранение артефактов для деплоя."""

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightgbm import LGBMClassifier

from src.config import MODEL_META_PATH, MODEL_PATH, PREPROCESSOR_PATH, RANDOM_STATE, project_root
from src.modeling import evaluate_binary, train_val_holdout, write_submission
from src.preprocessing import load_train_test, prepare_data

BEST_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.7,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
    "force_row_wise": True,
}


def main():
    base = project_root()
    raw = base / "data" / "raw"
    proc = base / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_train_test(raw)
    X, y, X_kaggle, kaggle_id, preprocessor = prepare_data(train, test)

    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_holdout(X, y, random_state=RANDOM_STATE)

    model = LGBMClassifier(**BEST_PARAMS)
    model.fit(X_tr, y_tr)

    val_metrics = evaluate_binary(model, X_va, y_va)
    test_metrics = evaluate_binary(model, X_te, y_te)

    model.fit(X, y)
    proba = model.predict_proba(X_kaggle)[:, 1]
    write_submission(kaggle_id, proba, proc / "submission.csv")

    joblib.dump(model, MODEL_PATH)
    preprocessor.save(PREPROCESSOR_PATH)

    meta = {
        "model": "LightGBM",
        "params": {k: v for k, v in BEST_PARAMS.items() if k not in ("verbosity", "force_row_wise")},
        "metrics": {
            "val_logloss": val_metrics["logloss"],
            "val_roc_auc": val_metrics["roc_auc"],
            "test_logloss": test_metrics["logloss"],
            "test_roc_auc": test_metrics["roc_auc"],
        },
        "n_features": int(X.shape[1]),
        "random_state": RANDOM_STATE,
    }
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Модель сохранена:", MODEL_PATH)
    print("Препроцессор:", PREPROCESSOR_PATH)
    print("Метрики val:", val_metrics)
    print("Метрики test:", test_metrics)
    print("Сабмит:", proc / "submission.csv")


if __name__ == "__main__":
    main()
