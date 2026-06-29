"""
Endpoints para entrenamiento de modelos
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, List
from pydantic import BaseModel
from api.tasks.celery_tasks import train_model_task

from api.services.trainer import training_service

router = APIRouter(prefix="/train", tags=["training"])


class TrainRequest(BaseModel):
    """Request para iniciar entrenamiento"""
    model_type: str  # "crn", "edct", "ct", "gnet", "rmsn", "msm"
    overrides: Optional[List[str]] = []
    force_cpu: bool = False


class TrainResponse(BaseModel):
    """Response para inicio de entrenamiento"""
    job_id: str
    status: str = "started"
    message: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Estado de un trabajo"""
    job_id: str
    status: str  # "running", "finished", "failed", "unknown"
    run_dir: Optional[str] = None
    model_type: Optional[str] = None
    return_code: Optional[int] = None
    error: Optional[str] = None


class RunsListResponse(BaseModel):
    """Lista de runs"""
    runs: List[dict]
    total: int


@router.post("/", response_model=TrainResponse)
async def start_training(request: TrainRequest):
    job_id = training_service.start_training(
            model_type=request.model_type,
            overrides=request.overrides,
            force_cpu=request.force_cpu
        )
    # Llamar a Celery
    task = train_model_task.delay(request.model_type, request.overrides, job_id)
    return TrainResponse(job_id=job_id, status="started", celery_task_id=task.id)


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_training_status(job_id: str):
    """
    Obtiene el estado de un entrenamiento
    """
    status = training_service.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} no encontrado")
    
    return JobStatusResponse(
        job_id=status.get("job_id", job_id),
        status=status.get("status", "unknown"),
        run_dir=status.get("run_dir"),
        model_type=status.get("model_type"),
        return_code=status.get("return_code")
    )


@router.get("/{job_id}/config")
async def get_training_config(job_id: str):
    """
    Obtiene la configuración completa de un entrenamiento
    """
    config = training_service.get_config(job_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} no encontrado")
    
    return config


@router.get("/runs", response_model=RunsListResponse)
async def list_runs(
    limit: int = 10,
    model_type: Optional[str] = None
):
    """
    Lista los entrenamientos realizados
    """
    runs = training_service.list_runs(limit=limit, model_type=model_type)
    return RunsListResponse(
        runs=runs,
        total=len(runs)
    )