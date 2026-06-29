"""
Tareas de Celery para entrenamiento asíncrono
"""
import logging
from celery import Celery
from celery.result import AsyncResult

from api.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND
from api.services.trainer import training_service

logger = logging.getLogger(__name__)

# Inicializar Celery
app = Celery(
    'causal_transformer_tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['api.tasks.celery_tasks']
)

# Configuración de Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 24,  # 24 horas máximo
    task_soft_time_limit=3600 * 23,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


@app.task(bind=True, name='train_model')
def train_model_task(self, model_type: str, overrides: list, job_id: str = None):
    """
    Tarea de Celery para entrenar un modelo
    
    Args:
        model_type: Tipo de modelo
        overrides: Overrides de Hydra
        job_id: ID opcional para el trabajo
    
    Returns:
        Dict con resultados
    """
    logger.info(f"Tarea de entrenamiento iniciada: {model_type}, job_id: {job_id}")
    
    try:
        # Iniciar entrenamiento
        job_id = training_service.start_training(
            model_type=model_type,
            overrides=overrides,
            job_id=job_id
        )
        
        # Actualizar estado de la tarea
        self.update_state(
            state='RUNNING',
            meta={'job_id': job_id, 'status': 'training_started'}
        )
        
        # Esperar a que termine (el proceso se ejecuta en background)
        # Nota: en producción, se podría hacer polling o usar callbacks
        import time
        while True:
            status = training_service.get_status(job_id)
            if not status:
                break
            if status.get('status') in ['finished', 'failed']:
                break
            time.sleep(10)
        
        # Obtener resultado final
        final_status = training_service.get_status(job_id)
        
        return {
            'job_id': job_id,
            'status': final_status.get('status', 'unknown'),
            'run_dir': final_status.get('run_dir'),
            'return_code': final_status.get('return_code')
        }
        
    except Exception as e:
        logger.error(f"Error en entrenamiento {job_id}: {e}")
        return {
            'job_id': job_id,
            'status': 'failed',
            'error': str(e)
        }


@app.task(name='get_task_status')
def get_task_status(task_id: str):
    """
    Obtiene el estado de una tarea de Celery
    
    Args:
        task_id: ID de la tarea
    
    Returns:
        Dict con el estado
    """
    result = AsyncResult(task_id, app=app)
    return {
        'task_id': task_id,
        'state': result.state,
        'result': result.result if result.ready() else None,
        'info': result.info if result.state == 'RUNNING' else None
    }