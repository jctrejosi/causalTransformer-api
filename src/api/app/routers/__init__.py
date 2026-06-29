# Contents of the file: /api/app/routers/__init__.py

from fastapi import APIRouter

router = APIRouter()

from .train import router as train_router
from .predict import router as predict_router
from .evaluate import router as evaluate_router

router.include_router(train_router, prefix="/train", tags=["train"])
router.include_router(predict_router, prefix="/predict", tags=["predict"])
router.include_router(evaluate_router, prefix="/evaluate", tags=["evaluate"])