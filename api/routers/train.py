"""
Endpoints para entrenamiento de modelos
"""
from fastapi import APIRouter, HTTPException
from typing import Optional, List
from pydantic import BaseModel

from api.services.trainer import training_service, generate_job_id
from api.tasks.celery_tasks import train_model_task
from celery.result import AsyncResult
from api.tasks.celery_tasks import app

router = APIRouter(prefix="/train", tags=["training"])


class TrainRequest(BaseModel):
    """Request para iniciar entrenamiento"""
    model_type: str  # "crn", "edct", "ct", "gnet", "rmsn", "msm"
    overrides: Optional[List[str]] = []
    force_cpu: bool = False


class TrainResponse(BaseModel):
    """Response para inicio de entrenamiento"""
    job_id: str
    celery_task_id: str
    status: str = "started"
    message: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Estado de un trabajo"""
    job_id: str
    celery_task_id: Optional[str] = None
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
    """
    Inicia un entrenamiento de un modelo causal usando Celery
    
    Ejemplos de overrides:
    - dataset.num_patients.train=512
    - exp.max_epochs=50
    - exp.gpus=1
    - model.encoder.br_size=64
    """
    try:
        # Generar job_id
        job_id = generate_job_id()
        
        # Llamar a Celery
        task = train_model_task.delay(
            model_type=request.model_type,
            overrides=request.overrides,
            job_id=job_id,
            force_cpu=request.force_cpu
        )
        
        return TrainResponse(
            job_id=job_id,
            celery_task_id=task.id,
            status="started",
            message=f"Entrenamiento de {request.model_type} enviado a Celery (task_id: {task.id})"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
        celery_task_id=status.get("celery_task_id"),
        status=status.get("status", "unknown"),
        run_dir=status.get("run_dir"),
        model_type=status.get("model_type"),
        return_code=status.get("return_code"),
        error=status.get("error")
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


# ======================== ENDPOINTS DE CELERY ========================

@router.get("/celery/{task_id}/status")
async def get_celery_task_status(task_id: str):
    """
    Obtiene el estado de una tarea de Celery
    
    Estados posibles:
    - PENDING: Esperando ser ejecutada
    - STARTED: Comenzó a ejecutarse
    - RUNNING: En ejecución
    - SUCCESS: Completada exitosamente
    - FAILURE: Falló
    - RETRY: Reintentando
    - REVOKED: Cancelada
    """
    result = AsyncResult(task_id, app=app)
    
    # Obtener información adicional si está en RUNNING
    info = None
    if result.state == "RUNNING" and result.info:
        info = result.info
    
    return {
        "task_id": task_id,
        "state": result.state,
        "result": result.result if result.ready() else None,
        "info": info,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
        "failed": result.failed() if result.ready() else None,
    }


@router.post("/celery/{task_id}/cancel")
async def cancel_celery_task(task_id: str):
    """
    Cancela una tarea de Celery en ejecución
    """
    from api.tasks.celery_tasks import cancel_task
    
    result = cancel_task.delay(task_id)
    return {
        "task_id": task_id,
        "status": "cancellation_requested",
        "celery_task_id": result.id
    }


@router.get("/celery/ping")
async def ping_celery():
    """
    Health check para Celery
    """
    from api.tasks.celery_tasks import ping_worker
    
    try:
        result = ping_worker.delay()
        # Esperar un poco a que el worker responda
        import time
        time.sleep(1)
        
        # Verificar si el worker respondió
        if result.ready():
            return {"status": "ok", "message": "Celery worker is alive", "result": result.result}
        else:
            return {"status": "ok", "message": "Celery worker received ping", "task_id": result.id}
    except Exception as e:
        return {"status": "error", "message": f"Celery worker not responding: {str(e)}"}