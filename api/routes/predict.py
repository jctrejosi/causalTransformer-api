"""
Endpoints para predicciones
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import numpy as np

from api.services.predictor import get_predictor

router = APIRouter(prefix="/predict", tags=["prediction"])


class PredictRequest(BaseModel):
    """Request para predicción"""
    run_id: str
    model_type: str
    submodel: Optional[str] = None
    data: Dict[str, Any]
    n_steps: Optional[int] = 1


class PredictResponse(BaseModel):
    """Response para predicción"""
    predictions: List[List[List[float]]]  # batch, time, features
    shape: List[int]


class PredictOneStepRequest(PredictRequest):
    """Request para one-step prediction"""
    pass


class PredictNStepRequest(PredictRequest):
    """Request para multi-step prediction"""
    n_steps: int


@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Realiza predicción con un modelo entrenado    """
    try:
        predictor = get_predictor(request.run_id, request.model_type, request.submodel)
        
        if request.n_steps > 1:
            predictions = predictor.predict_n_step(request.data, request.n_steps)
        else:
            predictions = predictor.predict(request.data)
        
        return PredictResponse(
            predictions=predictions.tolist(),
            shape=list(predictions.shape)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/one-step", response_model=PredictResponse)
async def predict_one_step(request: PredictOneStepRequest):
    """
    Realiza predicción one-step-ahead
    """
    try:
        predictor = get_predictor(request.run_id, request.model_type, request.submodel)
        predictions = predictor.predict_one_step(request.data)
        
        return PredictResponse(
            predictions=predictions.tolist(),
            shape=list(predictions.shape)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/n-step", response_model=PredictResponse)
async def predict_n_step(request: PredictNStepRequest):
    """
    Realiza predicción multi-step
    """
    try:
        predictor = get_predictor(request.run_id, request.model_type, request.submodel)
        predictions = predictor.predict_n_step(request.data, request.n_steps)
        
        return PredictResponse(
            predictions=predictions.tolist(),
            shape=list(predictions.shape)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health")
async def predict_health():
    """Health check para el servicio de predicción"""
    return {"status": "ok", "message": "Prediction service is running"}