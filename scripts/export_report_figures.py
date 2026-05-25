#!/usr/bin/env python3
"""Генерация графиков для отчёта."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightgbm import LGBMClassifier

from src.config import MODEL_PATH, RANDOM_STATE, project_root
from src.modeling import train_val_holdout
from src.preprocessing import load_train_test, prepare_data

sns.set_theme(style="whitegrid")


def main():
    base = project_root()
    fig_dir = base / "report" / "images"
    fig_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_train_test(base / "data" / "raw")
    id_col = "ID" if "ID" in train.columns else "id"
    feat_cols = [c for c in train.columns if c not in (id_col, "target")]

    plt.figure(figsize=(5, 4))
    train["target"].value_counts(normalize=True).sort_index().plot(kind="bar", color=["#4C72B0", "#DD8452"])
    plt.title("Распределение target")
    plt.xlabel("target")
    plt.ylabel("доля")
    plt.tight_layout()
    plt.savefig(fig_dir / "target_distribution.png", dpi=120)
    plt.close()

    miss = train[feat_cols].isna().mean().sort_values(ascending=False).head(20)
    plt.figure(figsize=(8, 5))
    miss.plot(kind="bar")
    plt.title("Топ-20 признаков по доле пропусков")
    plt.ylabel("доля NaN")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "missing_top20.png", dpi=120)
    plt.close()

    X, y, _, _, _ = prepare_data(train, test)
    X_tr, X_va, _, y_tr, y_va, _ = train_val_holdout(X, y, random_state=RANDOM_STATE)

    pca = PCA(n_components=50, random_state=RANDOM_STATE)
    pca.fit(X_tr)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 51), pca.explained_variance_ratio_, marker="o")
    plt.title("PCA: explained variance ratio (50 компонент)")
    plt.xlabel("компонента")
    plt.ylabel("доля дисперсии")
    plt.tight_layout()
    plt.savefig(fig_dir / "pca_variance.png", dpi=120)
    plt.close()

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_tr, y_tr)
    coef = pd.Series(lr.coef_[0], index=X.columns).abs().sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 5))
    coef.sort_values().plot(kind="barh")
    plt.title("Logistic Regression: |коэффициенты| (top 15)")
    plt.tight_layout()
    plt.savefig(fig_dir / "lr_coefficients.png", dpi=120)
    plt.close()

    # model comparison bar chart from notebook experiments
    models = {
        "LightGBM": 0.472561,
        "GradientBoosting": 0.477184,
        "RandomForest": 0.484564,
        "LogisticRegression": 0.498149,
        "LR baseline (без FE)": 0.497603,
    }
    plt.figure(figsize=(7, 4))
    pd.Series(models).sort_values().plot(kind="barh", color="#55A868")
    plt.title("Сравнение моделей (Log Loss на val)")
    plt.xlabel("Log Loss")
    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison.png", dpi=120)
    plt.close()

    if MODEL_PATH.exists():
        import joblib

        model = joblib.load(MODEL_PATH)
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(25)
        plt.figure(figsize=(8, 8))
        imp.sort_values().plot(kind="barh")
        plt.title("LightGBM feature importance (top 25)")
        plt.xlabel("importance")
        plt.tight_layout()
        plt.savefig(fig_dir / "lgbm_feature_importance.png", dpi=120)
        plt.close()

    print("Графики сохранены в", fig_dir)


if __name__ == "__main__":
    main()
