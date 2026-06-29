"""
Carga de modelos entrenados
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
import torch
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf, DictConfig
import logging

from api.config import MODELS_DIR, MODEL_CLASS_MAP, MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

# Cache de modelos cargados
_model_cache = {}


def load_model_from_run(
    run_id: str,
    model_type: str,
    submodel: Optional[str] = None,
    use_cache: bool = True
) -> Tuple[torch.nn.Module, DictConfig]:
    """
    Carga un modelo desde una run de MLflow
    
    Args:
        run_id: ID de la run en MLflow
        model_type: Tipo de modelo
        submodel: Submodelo a cargar (encoder, decoder, etc.)
        use_cache: Usar caché
    
    Returns:
        Tuple de (modelo, configuración)
    """
    cache_key = f"{run_id}_{model_type}_{submodel or 'full'}"
    
    if use_cache and cache_key in _model_cache:
        logger.info(f"Modelo {cache_key} cargado desde caché")
        return _model_cache[cache_key]
    
    # Conectar a MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Obtener información de la run
    run = client.get_run(run_id)
    if not run:
        raise ValueError(f"Run {run_id} no encontrada en MLflow")
    
    # Descargar artefactos a un directorio temporal
    temp_dir = tempfile.mkdtemp(prefix=f"model_{run_id}_")
    
    try:
        # Descargar configuración
        config_local = client.download_artifacts(run_id, ".hydra/config.yaml", dst_path=temp_dir)
        config_path = Path(config_local)
        config = OmegaConf.load(config_path)
        
        # Descargar checkpoints
        checkpoints_dir = Path(temp_dir) / "checkpoints"
        client.download_artifacts(run_id, "checkpoints", dst_path=temp_dir)
        
        # Buscar el checkpoint más reciente
        checkpoints = list(checkpoints_dir.glob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No se encontraron checkpoints para run {run_id}")
        
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        checkpoint_path = checkpoints[0]
        
        # Construir el nombre de la clase del modelo
        model_class_map = MODEL_CLASS_MAP.get(model_type, {})
        class_path = None
        
        if submodel:
            class_path = model_class_map.get(submodel)
        else:
            # Para single-stage models (CT, GNet) o tomar el primero disponible
            if "multi" in model_class_map:
                class_path = model_class_map["multi"]
            elif "g_net" in model_class_map:
                class_path = model_class_map["g_net"]
            elif "msm_regressor" in model_class_map:
                class_path = model_class_map["msm_regressor"]
            else:
                # Para encoder-decoder, cargar encoder por defecto
                class_path = model_class_map.get("encoder")
        
        if not class_path:
            raise ValueError(f"No se pudo determinar la clase para modelo {model_type}")
        
        # Importar dinámicamente la clase
        module_path, class_name = class_path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # Cargar modelo desde checkpoint
        # Nota: para LightningModule, usamos load_from_checkpoint
        model = model_class.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            args=config,
            map_location=torch.device("cpu"),
            strict=False
        )
        model.eval()
        
        # Guardar en caché
        _model_cache[cache_key] = (model, config)
        
        logger.info(f"Modelo {cache_key} cargado exitosamente desde {checkpoint_path}")
        return model, config
        
    except Exception as e:
        logger.error(f"Error cargando modelo {run_id}: {e}")
        # Limpiar directorio temporal
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def load_model_from_path(
    run_dir: Path,
    model_type: str,
    submodel: Optional[str] = None,
    use_cache: bool = True
) -> Tuple[torch.nn.Module, DictConfig]:
    """
    Carga un modelo desde un directorio local
    
    Args:
        run_dir: Directorio de la run
        model_type: Tipo de modelo
        submodel: Submodelo a cargar
        use_cache: Usar caché
    
    Returns:
        Tuple de (modelo, configuración)
    """
    cache_key = f"local_{run_dir}_{model_type}_{submodel or 'full'}"
    
    if use_cache and cache_key in _model_cache:
        logger.info(f"Modelo {cache_key} cargado desde caché")
        return _model_cache[cache_key]
    
    run_dir = Path(run_dir)
    
    # Cargar configuración
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuración no encontrada en {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Buscar checkpoints
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Directorios de checkpoints no encontrado en {checkpoints_dir}")
    
    checkpoints = list(checkpoints_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No se encontraron checkpoints en {checkpoints_dir}")
    
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    checkpoint_path = checkpoints[0]
    
    # Obtener clase del modelo
    model_class_map = MODEL_CLASS_MAP.get(model_type, {})
    class_path = None
    
    if submodel:
        class_path = model_class_map.get(submodel)
    else:
        if "multi" in model_class_map:
            class_path = model_class_map["multi"]
        elif "g_net" in model_class_map:
            class_path = model_class_map["g_net"]
        elif "msm_regressor" in model_class_map:
            class_path = model_class_map["msm_regressor"]
        else:
            class_path = model_class_map.get("encoder")
    
    if not class_path:
        raise ValueError(f"No se pudo determinar la clase para modelo {model_type}")
    
    # Importar y cargar
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    model = model_class.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        args=config,
        map_location=torch.device("cpu"),
        strict=False
    )
    model.eval()
    
    _model_cache[cache_key] = (model, config)
    return model, config


def clear_model_cache():
    """Limpia la caché de modelos"""
    _model_cache.clear()
    logger.info("Caché de modelos limpiada")