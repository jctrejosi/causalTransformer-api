"""
Schemas para requests
"""
from pydantic import BaseModel
from typing import Dict, Any, List, Optional


class TrainRequest(BaseModel):
    """Request para iniciar entrenamiento"""
    model_type: str
    overrides: Optional[List[str]] = []
    force_cpu: bool = False


class PredictRequest(BaseModel):
    """Request base para predicción"""
    run_id: str
    model_type: str
    submodel: Optional[str] = None
    data: Dict[str, Any]
    n_steps: Optional[int] = 1


class PredictOneStepRequest(PredictRequest):
    """Request para one-step prediction"""
    pass


class PredictNStepRequest(PredictRequest):
    """Request para multi-step prediction"""
    n_steps: int