"""
Schemas para responses
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class TrainResponse(BaseModel):
    """Response para inicio de entrenamiento"""
    job_id: str
    celery_task_id: str
    status: str = "started"
    message: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Estado de un trabajo"""
    job_id: str
    status: str
    run_dir: Optional[str] = None
    model_type: Optional[str] = None
    return_code: Optional[int] = None
    error: Optional[str] = None


class RunsListResponse(BaseModel):
    """Lista de runs"""
    runs: List[Dict[str, Any]]
    total: int


class PredictResponse(BaseModel):
    """Response para predicción"""
    predictions: List[List[List[float]]]
    shape: List[int]