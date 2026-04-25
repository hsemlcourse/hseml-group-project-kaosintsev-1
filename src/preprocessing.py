from __future__ import annotations

from pathlib import Path

import pandas as pd


def resolve_id_column(df: pd.DataFrame) -> str:
    if "id" in df.columns:
        return "id"
    if "ID" in df.columns:
        return "ID"
    raise KeyError("ID column not found")


def load_train_test(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test
