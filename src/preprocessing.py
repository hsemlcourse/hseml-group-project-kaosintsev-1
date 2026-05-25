import pickle
from dataclasses import dataclass
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


@dataclass
class Preprocessor:
    feat_cols: list
    num_cols: list
    cat_cols: list
    medians: pd.Series
    encoders: dict

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def fit(cls, train, test):
        train = train.copy()
        test = test.copy()

        id_col = resolve_id_column(train)
        train = train.drop_duplicates(subset=[id_col]).reset_index(drop=True)
        test = test.drop_duplicates(subset=[id_col]).reset_index(drop=True)

        feat_cols = [c for c in train.columns if c not in (id_col, "target")]
        num_cols = train[feat_cols].select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in feat_cols if c not in num_cols]

        medians = train[num_cols].median()
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            both = pd.concat([train[col], test[col]], axis=0).astype(str).fillna(MISSING_TOKEN)
            le.fit(both)
            encoders[col] = le

        return cls(
            feat_cols=feat_cols,
            num_cols=num_cols,
            cat_cols=cat_cols,
            medians=medians,
            encoders=encoders,
        )

    def transform(self, df):
        df = df.copy()
        raw = df[self.feat_cols].copy()

        for col in self.num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[self.num_cols] = df[self.num_cols].fillna(self.medians)
        for col in self.cat_cols:
            df[col] = df[col].fillna(MISSING_TOKEN).astype(str)
            le = self.encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else MISSING_TOKEN)
            df[col] = le.transform(df[col])

        out = df[self.feat_cols].copy()
        out["count_nan_per_row"] = raw.isna().sum(axis=1)
        out["count_cat_per_row"] = raw[self.cat_cols].notna().sum(axis=1)
        out["row_num_mean"] = df[self.num_cols].mean(axis=1)
        out["row_num_std"] = df[self.num_cols].std(axis=1)
        out["row_num_min"] = df[self.num_cols].min(axis=1)
        out["row_num_max"] = df[self.num_cols].max(axis=1)
        return out


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

    preprocessor = Preprocessor.fit(train, test)
    X_train = preprocessor.transform(train)
    y = train["target"].astype(int)

    X_kaggle = preprocessor.transform(test)
    kaggle_id = test[id_col]

    return X_train, y, X_kaggle, kaggle_id, preprocessor


def prepare_from_paths(train_path, test_path, save_preprocessor=None):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, y, X_kaggle, kaggle_id, preprocessor = prepare_data(train, test)
    if save_preprocessor is not None:
        preprocessor.save(save_preprocessor)
    return X_train, y, X_kaggle, kaggle_id
