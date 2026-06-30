"""
Servicios de la API
"""
from api.services.trainer import TrainingService, training_service
from api.services.predictor import Predictor, get_predictor
from api.services.evaluator import Evaluator, get_evaluator