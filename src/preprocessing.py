import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import MISSING_TOKEN


def resolve_id_column(df):
    if "id" in df.columns:
        return "id"
    if "ID" in df.columns:
        return "ID"
    raise KeyError("нет колонки id")


def load_train_test(data_dir):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def prepare_data(train, test):
    train = train.copy()
    test = test.copy()

    id_col = resolve_id_column(train)

    train = train.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    test = test.drop_duplicates(subset=[id_col]).reset_index(drop=True)

    feat_cols = [c for c in train.columns if c not in (id_col, "target")]
    num_cols = train[feat_cols].select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in feat_cols if c not in num_cols]

    train_raw = train[feat_cols].copy()
    test_raw = test[feat_cols].copy()

    med = train[num_cols].median()
    train[num_cols] = train[num_cols].fillna(med)
    test[num_cols] = test[num_cols].fillna(med)

    for c in cat_cols:
        train[c] = train[c].fillna(MISSING_TOKEN)
        test[c] = test[c].fillna(MISSING_TOKEN)

    for c in cat_cols:
        le = LabelEncoder()
        both = pd.concat([train[c], test[c]], axis=0).astype(str)
        le.fit(both)
        train[c] = le.transform(train[c].astype(str))
        test[c] = le.transform(test[c].astype(str))

    X_train = train[feat_cols].copy()
    X_train["count_nan_per_row"] = train_raw.isna().sum(axis=1)
    X_train["count_cat_per_row"] = train_raw[cat_cols].notna().sum(axis=1)
    X_train["row_num_mean"] = train[num_cols].mean(axis=1)
    X_train["row_num_std"] = train[num_cols].std(axis=1)
    X_train["row_num_min"] = train[num_cols].min(axis=1)
    X_train["row_num_max"] = train[num_cols].max(axis=1)

    y = train["target"].astype(int)

    X_kaggle = test[feat_cols].copy()
    X_kaggle["count_nan_per_row"] = test_raw.isna().sum(axis=1)
    X_kaggle["count_cat_per_row"] = test_raw[cat_cols].notna().sum(axis=1)
    X_kaggle["row_num_mean"] = test[num_cols].mean(axis=1)
    X_kaggle["row_num_std"] = test[num_cols].std(axis=1)
    X_kaggle["row_num_min"] = test[num_cols].min(axis=1)
    X_kaggle["row_num_max"] = test[num_cols].max(axis=1)

    kaggle_id = test[id_col]

    return X_train, y, X_kaggle, kaggle_id


def prepare_from_paths(train_path, test_path, save_preprocessor=None):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, y, X_kaggle, kaggle_id = prepare_data(train, test)
    if save_preprocessor is not None:
        Path(save_preprocessor).parent.mkdir(parents=True, exist_ok=True)
        with open(save_preprocessor, "wb") as f:
            pickle.dump({"train_csv": str(train_path), "test_csv": str(test_path)}, f)
    return X_train, y, X_kaggle, kaggle_id
