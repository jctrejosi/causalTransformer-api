"""
Punto de entrada de la API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from api.routers import train, predict, evaluate
from api.config import STORAGE_DIR

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear la aplicación FastAPI
app = FastAPI(
    title="Causal Transformer API",
    description="API para entrenamiento y predicción de modelos causales (CRN, EDCT, CT, G-Net, RMSN, MSM)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar routers
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(evaluate.router)


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "name": "Causal Transformer API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "storage": str(STORAGE_DIR)
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error inesperado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor", "message": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )