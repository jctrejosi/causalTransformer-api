#!/usr/bin/env python
"""
Script para iniciar la API
"""
import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Configurar variables de entorno
os.environ["PYTHONPATH"] = str(ROOT_DIR)

if __name__ == "__main__":
    import uvicorn
    
    # Asegurar que los directorios existan
    from api.config import STORAGE_DIR
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Iniciando API Causal Transformer...")
    print(f"📁 Storage dir: {STORAGE_DIR}")
    print(f"📚 Documentación: http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )