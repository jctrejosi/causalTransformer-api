"""
Dependencias compartidas de la API
"""
from functools import lru_cache
from typing import Optional

from api.config import STORAGE_DIR, MLFLOW_TRACKING_URI
from api.services.trainer import TrainingService


@lru_cache()
def get_training_service() -> TrainingService:
    """Obtiene la instancia del servicio de entrenamiento"""
    return TrainingService(runs_dir=STORAGE_DIR / "runs")


# No tenemos dependencias de BD por ahora