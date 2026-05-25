import json
import os

import pandas as pd
import requests
import streamlit as st

from src.config import project_root
from src.inference import get_predictor
from src.utils import sanitize_features

API_URL = os.getenv("API_URL", "http://api:8000")
DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 1000


def predict_via_api(features: dict) -> dict:
    clean = sanitize_features(features)
    resp = requests.post(f"{API_URL}/predict", json={"features": clean}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def predict_batch_via_api(records: list[dict]) -> list[dict]:
    all_results = []
    for start in range(0, len(records), BATCH_SIZE):
        chunk = [sanitize_features(record) for record in records[start : start + BATCH_SIZE]]
        resp = requests.post(
            f"{API_URL}/predict/batch",
            json={"records": chunk},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        for proba, pred in zip(data["probabilities"], data["predictions"]):
            all_results.append(
                {
                    "probability": proba,
                    "prediction": pred,
                    "threshold": data["threshold"],
                }
            )
    return all_results


def predict_local(features: dict) -> dict:
    proba = get_predictor().predict_proba(sanitize_features(features))
    return {
        "probability": round(proba, 6),
        "prediction": int(proba >= DEFAULT_THRESHOLD),
        "threshold": DEFAULT_THRESHOLD,
    }


def load_sample_row():
    test_path = project_root() / "data" / "raw" / "test.csv"
    if not test_path.exists():
        return None
    test = pd.read_csv(test_path).head(1)
    id_col = "ID" if "ID" in test.columns else "id"
    feat_cols = [c for c in test.columns if c not in (id_col, "target")]
    return test.iloc[0][feat_cols].to_dict()


st.set_page_config(page_title="Claims Predictor", page_icon="📋", layout="wide")
st.title("BNP Paribas — предсказание одобрения заявки")
st.caption("Бинарная классификация: вероятность быстрого одобрения страховой заявки")

mode = st.sidebar.radio("Режим", ["Локальная модель", "Через FastAPI"])
use_api = mode == "Через FastAPI"
if use_api:
    st.sidebar.caption(f"API: `{API_URL}`")

uploaded = st.file_uploader("Загрузить CSV с одной или несколькими заявками", type=["csv"])
sample = load_sample_row()

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Загруженные данные")
    st.dataframe(df.head(), use_container_width=True)

    id_col = "ID" if "ID" in df.columns else ("id" if "id" in df.columns else None)
    feat_cols = [c for c in df.columns if c not in (id_col, "target")]

    if st.button("Предсказать для всех строк"):
        try:
            if use_api:
                with st.spinner(f"Отправка {len(df)} заявок в API (батчами по {BATCH_SIZE})..."):
                    records = [row.to_dict() for _, row in df[feat_cols].iterrows()]
                    results = predict_batch_via_api(records)
            else:
                predictor = get_predictor()
                probas = predictor.predict_batch(df)
                results = [
                    {
                        "probability": round(p, 6),
                        "prediction": int(p >= DEFAULT_THRESHOLD),
                        "threshold": DEFAULT_THRESHOLD,
                    }
                    for p in probas
                ]
            out = df.copy()
            out["PredictedProb"] = [r["probability"] for r in results]
            out["Prediction"] = [r["prediction"] for r in results]
            st.success(f"Готово: {len(results)} предсказаний")
            st.dataframe(out, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))
else:
    st.subheader("Пример одной заявки")
    if sample is None:
        st.warning("Положите test.csv в data/raw/ или загрузите CSV.")
    else:
        st.json({k: sample[k] for k in list(sample.keys())[:8]} | {"...": "ещё признаки"})
        if st.button("Предсказать для примера"):
            try:
                result = predict_via_api(sample) if use_api else predict_local(sample)
                st.metric("Вероятность класса 1", f"{result['probability']:.4f}")
                label = "Быстрое одобрение" if result["prediction"] == 1 else "Нужна доп. проверка"
                st.info(f"Решение (порог {result['threshold']}): **{label}**")
            except Exception as exc:
                st.error(str(exc))

with st.expander("Метаданные модели"):
    meta_path = project_root() / "models" / "model_meta.json"
    if meta_path.exists():
        st.code(json.dumps(json.loads(meta_path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    else:
        st.write("Сначала обучите модель: `python scripts/train_model.py`")
