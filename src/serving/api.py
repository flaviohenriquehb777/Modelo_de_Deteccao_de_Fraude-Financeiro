"""Camada de serving de referencia (FastAPI) sobre o contrato de inferencia.

Este servico reaproveita o contrato oficial em ``src.serving.inference``
(``load_inference_bundle`` + ``predict_dataframe``) e expoe uma API HTTP
minima: ``/health``, ``/model/metadata`` e ``/predict``.

Importante: e uma camada de **referencia**, executada localmente ou via
``docker run``. Nao ha deploy, SLA, autenticacao nem monitoramento. A imagem
Docker existe para provar que o modelo empacota e sobe como servico.
"""

import json
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config.paths import MODEL_METADATA_PATH
from src.serving.inference import InferenceBundle, load_inference_bundle, predict_dataframe

app = FastAPI(
    title="Deteccao de Fraude - Serving de Referencia",
    description="API HTTP de referencia sobre o contrato de inferencia do modelo.",
    version="1.0.0",
)

# Limite de registros por requisicao para proteger o servico de payloads abusivos.
MAX_RECORDS_PER_REQUEST = 10_000


@lru_cache(maxsize=1)
def get_bundle() -> InferenceBundle:
    """Carrega (uma unica vez) o bundle versionado pelo DVC."""
    return load_inference_bundle()


class PredictRequest(BaseModel):
    """Lote de transacoes cruas; as features obrigatorias vem de /model/metadata."""

    records: list[dict[str, float]] = Field(
        ...,
        min_length=1,
        max_length=MAX_RECORDS_PER_REQUEST,
        description="Lista de transacoes; cada item deve conter as features obrigatorias.",
    )


class PredictionItem(BaseModel):
    fraud_probability: float
    fraud_prediction: int


class PredictResponse(BaseModel):
    count: int
    decision_threshold: float
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    decision_threshold: float
    n_features: int


class ModelMetadataResponse(BaseModel):
    decision_threshold: float
    n_features: int
    features: list[str]
    metrics: dict


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = get_bundle()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        decision_threshold=bundle.threshold,
        n_features=len(bundle.feature_names),
    )


@app.get("/model/metadata", response_model=ModelMetadataResponse)
def model_metadata() -> ModelMetadataResponse:
    bundle = get_bundle()
    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    metrics = {
        "out_of_fold": metadata.get("out_of_fold_calibrated_metrics", {}),
        "acceptance_passed": metadata.get("acceptance", {}).get("passed", False),
    }
    return ModelMetadataResponse(
        decision_threshold=bundle.threshold,
        n_features=len(bundle.feature_names),
        features=bundle.feature_names,
        metrics=metrics,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    bundle = get_bundle()
    features = pd.DataFrame(request.records)
    try:
        predictions = predict_dataframe(features, bundle)
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    items = [
        PredictionItem(
            fraud_probability=float(row.fraud_probability),
            fraud_prediction=int(row.fraud_prediction),
        )
        for row in predictions.itertuples(index=False)
    ]
    return PredictResponse(
        count=len(items),
        decision_threshold=bundle.threshold,
        predictions=items,
    )
