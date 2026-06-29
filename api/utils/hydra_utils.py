"""
Utilidades para trabajar con Hydra desde código
"""
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)

# Directorio del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent


def build_hydra_command(
    script_path: str,
    overrides: List[str],
    run_dir: Optional[Path] = None,
    config_name: str = "config.yaml",
    config_path: str = "../config/"
) -> List[str]:
    """
    Construye el comando para ejecutar un script de Hydra
    
    Args:
        script_path: Ruta al script (relativa a PROJECT_ROOT)
        overrides: Lista de overrides de Hydra
        run_dir: Directorio donde guardar los outputs
        config_name: Nombre del archivo de configuración
        config_path: Ruta a la carpeta de configuraciones
    
    Returns:
        Lista con el comando para subprocess
    """
    # Ruta absoluta al script
    full_script_path = PROJECT_ROOT / script_path
    
    # Verificar que el script existe
    if not full_script_path.exists():
        raise FileNotFoundError(f"Script no encontrado: {full_script_path}")
    
    cmd = [
        sys.executable,  # Usar el mismo Python
        str(full_script_path),
        f"hydra.run.dir={run_dir}" if run_dir else "",
        f"hydra.job.name={Path(script_path).stem}",
    ]
    
    # Agregar overrides
    cmd.extend(overrides)
    
    # Filtrar elementos vacíos
    cmd = [c for c in cmd if c]
    
    logger.info(f"Comando Hydra construido: {' '.join(cmd)}")
    return cmd


def run_hydra_script(
    script_path: str,
    overrides: List[str],
    run_dir: Optional[Path] = None,
    env_vars: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None
) -> subprocess.Popen:
    """
    Ejecuta un script de Hydra en un subproceso
    
    Args:
        script_path: Ruta al script
        overrides: Overrides de Hydra
        run_dir: Directorio de salida
        env_vars: Variables de entorno adicionales
        timeout: Timeout en segundos (opcional)
    
    Returns:
        subprocess.Popen: Proceso en ejecución
    """
    cmd = build_hydra_command(script_path, overrides, run_dir)
    
    # Preparar entorno
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    # Ejecutar en background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        cwd=PROJECT_ROOT
    )
    
    logger.info(f"Proceso iniciado con PID: {process.pid}")
    return process


def load_config_from_run(run_dir: Path) -> DictConfig:
    """
    Carga la configuración de una run de Hydra
    
    Args:
        run_dir: Directorio de la run
    
    Returns:
        DictConfig con la configuración
    """
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuración no encontrada en: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def save_training_metadata(
    job_id: str,
    model_type: str,
    overrides: List[str],
    process: subprocess.Popen,
    run_dir: Path,
    celery_task_id: Optional[str] = None  # <--- NUEVO PARÁMETRO
) -> Dict[str, Any]:
    """
    Guarda metadatos del entrenamiento
    
    Args:
        job_id: ID único del trabajo
        model_type: Tipo de modelo
        overrides: Overrides usados
        process: Proceso en ejecución
        run_dir: Directorio de la run
        celery_task_id: ID de la tarea de Celery
    
    Returns:
        Diccionario con metadatos
    """
    from datetime import datetime
    
    metadata = {
        "job_id": job_id,
        "model_type": model_type,
        "overrides": overrides,
        "pid": process.pid,
        "run_dir": str(run_dir),
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "celery_task_id": celery_task_id,  # <--- GUARDAR CELERY_TASK_ID
    }
    
    # Guardar en archivo
    metadata_path = run_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)
    
    return metadata


def get_job_status(job_id: str, runs_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Obtiene el estado de un trabajo
    
    Args:
        job_id: ID del trabajo
        runs_dir: Directorio donde se guardan las runs
    
    Returns:
        Diccionario con el estado o None si no existe
    """
    run_dir = runs_dir / job_id
    if not run_dir.exists():
        return None
    
    metadata_path = run_dir / "metadata.yaml"
    if not metadata_path.exists():
        return {"job_id": job_id, "status": "unknown", "run_dir": str(run_dir)}
    
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)
    
    # Verificar si el proceso sigue vivo
    if metadata.get("status") == "running":
        pid = metadata.get("pid")
        if pid:
            try:
                # Verificar si el proceso aún existe
                os.kill(pid, 0)
            except OSError:
                metadata["status"] = "finished"
                # Actualizar metadata
                with open(metadata_path, "w") as f:
                    yaml.dump(metadata, f)
        else:
            metadata["status"] = "unknown"
    
    return metadata


def get_model_checkpoint(run_dir: Path, model_type: str) -> Optional[Path]:
    """
    Encuentra el checkpoint del modelo en el directorio de la run
    
    Args:
        run_dir: Directorio de la run
        model_type: Tipo de modelo
    
    Returns:
        Path al checkpoint o None si no se encuentra
    """
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    
    # Buscar el último checkpoint
    checkpoints = list(checkpoints_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    
    # Ordenar por fecha de modificación
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0]