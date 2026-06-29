"""
Tareas asíncronas para la API
"""
from api.tasks.celery_tasks import app, train_model_task