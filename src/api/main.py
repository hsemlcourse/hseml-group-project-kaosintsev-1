from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import get_predictor
from src.utils import sanitize_features

app = FastAPI(
    title="BNP Paribas Claims API",
    description="API для предсказания вероятности быстрого одобрения страховой заявки",
    version="1.0.0",
)

DEFAULT_THRESHOLD = 0.5


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Словарь признаков v1..v131 и категориальных полей")


class PredictResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., description="0 — нужна проверка, 1 — быстрое одобрение")
    threshold: float = DEFAULT_THRESHOLD


class BatchPredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., description="Список заявок — каждая как словарь признаков")


class BatchPredictResponse(BaseModel):
    probabilities: list[float]
    predictions: list[int]
    threshold: float = DEFAULT_THRESHOLD


def _to_response(proba: float) -> PredictResponse:
    return PredictResponse(
        probability=round(proba, 6),
        prediction=int(proba >= DEFAULT_THRESHOLD),
        threshold=DEFAULT_THRESHOLD,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    predictor = get_predictor()
    return {
        "model_path": str(predictor.model_path),
        "meta": predictor.meta,
        "n_features": len(predictor.preprocessor.feat_cols) + 6,
        "feature_names": predictor.preprocessor.feat_cols[:10] + ["..."],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    try:
        proba = get_predictor().predict_proba(sanitize_features(body.features))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {exc}") from exc

    return _to_response(proba)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest):
    if not body.records:
        raise HTTPException(status_code=400, detail="records не может быть пустым")

    try:
        import pandas as pd

        predictor = get_predictor()
        df = pd.DataFrame([sanitize_features(record) for record in body.records])
        probas = predictor.predict_batch(df)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {exc}") from exc

    return BatchPredictResponse(
        probabilities=[round(float(p), 6) for p in probas],
        predictions=[int(p >= DEFAULT_THRESHOLD) for p in probas],
        threshold=DEFAULT_THRESHOLD,
    )
