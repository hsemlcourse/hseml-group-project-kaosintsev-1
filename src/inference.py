import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_META_PATH, MODEL_PATH, PREPROCESSOR_PATH
from src.preprocessing import Preprocessor


class ClaimPredictor:
    def __init__(self, model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH, meta_path=MODEL_META_PATH):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.meta_path = Path(meta_path)
        self.model = None
        self.preprocessor = None
        self.meta = {}

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {self.model_path}. Запустите: python scripts/train_model.py"
            )
        self.model = joblib.load(self.model_path)
        self.preprocessor = Preprocessor.load(self.preprocessor_path)
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return self

    def predict_proba(self, features: dict) -> float:
        if self.model is None or self.preprocessor is None:
            self.load()
        row = {col: features.get(col, None) for col in self.preprocessor.feat_cols}
        df = pd.DataFrame([row])
        X = self.preprocessor.transform(df)
        return float(self.model.predict_proba(X)[0, 1])

    def predict_batch(self, df: pd.DataFrame) -> list[float]:
        if self.model is None or self.preprocessor is None:
            self.load()
        X = self.preprocessor.transform(df)
        return self.model.predict_proba(X)[:, 1].tolist()


_predictor = None


def get_predictor() -> ClaimPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ClaimPredictor().load()
    return _predictor
