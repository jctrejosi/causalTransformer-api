"""
Servicio para gestionar entrenamientos
"""
import uuid
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from omegaconf import OmegaConf

from api.config import (
    RUNS_DIR, 
    MODEL_SCRIPT_MAP, 
    ENCODER_DECODER_MODELS,
    SINGLE_STAGE_MODELS
)
from api.utils.hydra_utils import (
    run_hydra_script,
    load_config_from_run,
    save_training_metadata,
    get_job_status as get_status_from_metadata
)

logger = logging.getLogger(__name__)


def generate_job_id() -> str:
    """Genera un ID único para el trabajo"""
    return f"job_{uuid.uuid4().hex[:12]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class TrainingService:
    """Servicio para gestionar entrenamientos de modelos causales"""
    
    def __init__(self, runs_dir: Path = RUNS_DIR):
        self.runs_dir = runs_dir
        self.active_training = {}  # job_id -> process
    
    def start_training(
        self, 
        model_type: str, 
        overrides: List[str],
        job_id: Optional[str] = None,
        force_cpu: bool = False
    ) -> str:
        """
        Inicia un entrenamiento
        
        Args:
            model_type: Tipo de modelo (crn, edct, ct, gnet, rmsn, msm)
            overrides: Lista de overrides de Hydra
            job_id: ID opcional para el trabajo
            force_cpu: Forzar uso de CPU
        
        Returns:
            job_id: ID del trabajo iniciado
        """
        # Validar modelo
        if model_type not in MODEL_SCRIPT_MAP:
            raise ValueError(f"Modelo no soportado: {model_type}. Opciones: {list(MODEL_SCRIPT_MAP.keys())}")
        
        # Generar job_id si no se proporciona
        if job_id is None:
            job_id = generate_job_id()
        
        # Preparar overrides
        full_overrides = overrides.copy()
        
        # Agregar overrides específicos si es necesario
        if force_cpu:
            full_overrides.append("exp.gpus=0")
        
        # Asegurar que el job_id se guarde en los metadatos
        full_overrides.append(f"exp.run_id={job_id}")
        
        # Directorio para esta run
        run_dir = self.runs_dir / job_id
        if run_dir.exists():
            logger.warning(f"El directorio {run_dir} ya existe, se sobrescribirá")
        
        # Determinar script a ejecutar
        script_path = MODEL_SCRIPT_MAP[model_type]
        
        # Iniciar proceso
        process = run_hydra_script(
            script_path=f"runnables/{script_path}",
            overrides=full_overrides,
            run_dir=run_dir
        )
        
        # Guardar metadatos
        metadata = save_training_metadata(job_id, model_type, full_overrides, process, run_dir)
        
        # Guardar proceso activo
        self.active_training[job_id] = process
        
        logger.info(f"Entrenamiento iniciado: {job_id} (modelo: {model_type})")
        return job_id
    
    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el estado de un entrenamiento
        
        Args:
            job_id: ID del trabajo
        
        Returns:
            Diccionario con el estado o None si no existe
        """
        status = get_status_from_metadata(job_id, self.runs_dir)
        
        # Verificar si el proceso aún está en active_training
        if status and status.get("status") == "running":
            process = self.active_training.get(job_id)
            if process:
                # Verificar si el proceso terminó
                poll_result = process.poll()
                if poll_result is not None:
                    status["status"] = "finished"
                    status["return_code"] = poll_result
                    # Limpiar proceso activo
                    del self.active_training[job_id]
        
        return status
    
    def get_config(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Carga la configuración de una run
        
        Args:
            job_id: ID del trabajo
        
        Returns:
            Configuración como diccionario o None si no existe
        """
        run_dir = self.runs_dir / job_id
        if not run_dir.exists():
            return None
        
        try:
            config = load_config_from_run(run_dir)
            return OmegaConf.to_container(config, resolve=True)
        except Exception as e:
            logger.error(f"Error cargando configuración de {job_id}: {e}")
            return None
    
    def list_runs(self, limit: int = 10, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lista las runs disponibles
        
        Args:
            limit: Número máximo de runs a listar
            model_type: Filtrar por tipo de modelo
        
        Returns:
            Lista de runs con metadatos
        """
        runs = []
        for run_dir in sorted(self.runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue
            
            metadata_path = run_dir / "metadata.yaml"
            if not metadata_path.exists():
                continue
            
            import yaml
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)
            
            # Filtrar por tipo
            if model_type and metadata.get("model_type") != model_type:
                continue
            
            runs.append(metadata)
            
            if len(runs) >= limit:
                break
        
        return runs


# Instancia global del servicio
training_service = TrainingService()