"""
Endpoints para evaluación de modelos
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import numpy as np

from api.services.evaulator import get_evaluator

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


class EvaluateRequest(BaseModel):
    """Request para evaluación"""
    run_id: str
    model_type: str
    submodel: Optional[str] = None
    test_data: Dict[str, Any]  # Datos con ground truth (outputs)


class EvaluateResponse(BaseModel):
    """Response para evaluación"""
    rmse_all: float
    rmse_orig: Optional[float] = None
    rmse_last: Optional[float] = None
    n_step_rmses: Optional[Dict[str, float]] = None
    metrics: Dict[str, float]


class EvaluateNStepRequest(BaseModel):
    """Request para evaluación multi-step"""
    run_id: str
    model_type: str
    submodel: Optional[str] = None
    test_data: Dict[str, Any]
    n_steps: int = 5


@router.post("/", response_model=EvaluateResponse)
async def evaluate_model(request: EvaluateRequest):
    """
    Evalúa un modelo entrenado contra datos de test
    """
    try:
        evaluator = get_evaluator(request.run_id, request.model_type, request.submodel)
        results = evaluator.evaluate(request.test_data)
        
        return EvaluateResponse(
            rmse_all=results.get('rmse_all', 0.0),
            rmse_orig=results.get('rmse_orig'),
            rmse_last=results.get('rmse_last'),
            n_step_rmses=results.get('n_step_rmses', {}),
            metrics=results.get('metrics', {})
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/n-step", response_model=EvaluateResponse)
async def evaluate_n_step(request: EvaluateNStepRequest):
    """
    Evalúa un modelo en predicción multi-step
    """
    try:
        evaluator = get_evaluator(request.run_id, request.model_type, request.submodel)
        results = evaluator.evaluate_n_step(request.test_data, request.n_steps)
        
        return EvaluateResponse(
            rmse_all=0.0,  # No aplica para n-step
            n_step_rmses=results.get('n_step_rmses', {}),
            metrics=results.get('metrics', {})
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))