"""
Tareas de Celery para entrenamiento asíncrono
"""
import logging
import time
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
    task_reject_on_worker_lost=True,
)


@app.task(bind=True, name='train_model', max_retries=3)
def train_model_task(self, model_type: str, overrides: list, job_id: str = None, force_cpu: bool = False):
    """
    Tarea de Celery para entrenar un modelo
    
    Args:
        model_type: Tipo de modelo
        overrides: Overrides de Hydra
        job_id: ID opcional para el trabajo
        force_cpu: Forzar uso de CPU
    
    Returns:
        Dict con resultados
    """
    logger.info(f"📊 Tarea de entrenamiento iniciada: {model_type}, job_id: {job_id}")
    
    try:
        # Iniciar entrenamiento
        job_id = training_service.start_training(
            model_type=model_type,
            overrides=overrides,
            job_id=job_id,
            force_cpu=force_cpu,
            celery_task_id=self.request.id  # Guardar el task_id de Celery
        )
        
        # Actualizar estado de la tarea
        self.update_state(
            state='RUNNING',
            meta={'job_id': job_id, 'status': 'training_started', 'message': 'Entrenamiento en progreso...'}
        )
        
        # Monitorear el progreso del entrenamiento
        last_status = None
        while True:
            status = training_service.get_status(job_id)
            if not status:
                time.sleep(5)
                continue
            
            current_status = status.get('status', 'unknown')
            
            if current_status != last_status:
                logger.info(f"📊 Estado actual: {current_status}")
                self.update_state(
                    state='RUNNING',
                    meta={'job_id': job_id, 'status': current_status}
                )
                last_status = current_status
            
            if current_status in ['finished', 'failed']:
                break
                
            time.sleep(10)
        
        # Obtener resultado final
        final_status = training_service.get_status(job_id)
        
        if final_status and final_status.get('status') == 'finished':
            return {
                'job_id': job_id,
                'status': 'finished',
                'run_dir': final_status.get('run_dir'),
                'return_code': final_status.get('return_code', 0),
                'message': 'Entrenamiento completado exitosamente'
            }
        else:
            error_msg = final_status.get('error', 'Entrenamiento fallido') if final_status else 'Entrenamiento fallido'
            return {
                'job_id': job_id,
                'status': 'failed',
                'run_dir': final_status.get('run_dir') if final_status else None,
                'error': error_msg
            }
        
    except Exception as e:
        logger.error(f"❌ Error en entrenamiento {job_id}: {e}", exc_info=True)
        
        if self.request.retries < self.max_retries:
            logger.info(f"Reintentando tarea {self.request.id} (intento {self.request.retries + 1})")
            self.retry(exc=e, countdown=60 * (self.request.retries + 1))
        
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
        'info': result.info if result.state == 'RUNNING' else None,
        'ready': result.ready(),
        'successful': result.successful() if result.ready() else None,
        'failed': result.failed() if result.ready() else None,
    }


@app.task(name='cancel_task')
def cancel_task(task_id: str):
    """
    Cancela una tarea de Celery
    
    Args:
        task_id: ID de la tarea
    """
    result = AsyncResult(task_id, app=app)
    result.revoke(terminate=True)
    return {'task_id': task_id, 'status': 'cancelled'}


@app.task(name='ping_worker')
def ping_worker():
    """Health check para Celery worker"""
    return {'status': 'ok', 'message': 'Celery worker is alive'}