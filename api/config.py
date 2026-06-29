"""
Configuración central de la API
"""
import os
from pathlib import Path
from typing import Optional

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent.parent.parent
API_DIR = Path(__file__).parent
STORAGE_DIR = API_DIR / "storage"
MODELS_DIR = STORAGE_DIR / "models"
RUNS_DIR = STORAGE_DIR / "runs"
TEMP_DIR = STORAGE_DIR / "temp"

# Crear directorios si no existen
for dir_path in [STORAGE_DIR, MODELS_DIR, RUNS_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuración de MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_PREFIX = os.getenv("MLFLOW_EXPERIMENT_PREFIX", "api")

# Configuración de Celery (opcional)
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Mapeo de tipos de modelo a scripts de entrenamiento
MODEL_SCRIPT_MAP = {
    "crn": "train_enc_dec.py",
    "edct": "train_enc_dec.py",
    "ct": "train_multi.py",
    "gnet": "train_gnet.py",
    "rmsn": "train_rmsn.py",
    "msm": "train_msm.py"
}

# Modelos que usan encoder-decoder (requieren entrenamiento secuencial)
ENCODER_DECODER_MODELS = {"crn", "edct", "rmsn"}

# Modelos que son single-stage
SINGLE_STAGE_MODELS = {"ct", "gnet", "msm"}

# Mapeo de modelo a su clase (para carga)
MODEL_CLASS_MAP = {
    "crn": {
        "encoder": "src.models.crn.CRNEncoder",
        "decoder": "src.models.crn.CRNDecoder"
    },
    "edct": {
        "encoder": "src.models.edct.EDCTEncoder",
        "decoder": "src.models.edct.EDCTDecoder"
    },
    "ct": {
        "multi": "src.models.ct.CT"
    },
    "gnet": {
        "g_net": "src.models.gnet.GNet"
    },
    "rmsn": {
        "encoder": "src.models.rmsn.RMSNEncoder",
        "decoder": "src.models.rmsn.RMSNDecoder",
        "propensity_treatment": "src.models.rmsn.RMSNPropensityNetworkTreatment",
        "propensity_history": "src.models.rmsn.RMSNPropensityNetworkHistory"
    },
    "msm": {
        "msm_regressor": "src.models.msm.MSMRegressor",
        "propensity_treatment": "src.models.msm.MSMPropensityTreatment",
        "propensity_history": "src.models.msm.MSMPropensityHistory"
    }
}

MAX_CONCURRENT_TRAININGS = int(os.getenv("MAX_CONCURRENT_TRAININGS", "2"))
DEFAULT_TRAIN_TIMEOUT = int(os.getenv("DEFAULT_TRAIN_TIMEOUT", "86400"))  # 24h