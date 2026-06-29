"""
Servicios de la API
"""
from api.services.trainer import TrainingService, training_service
from api.services.predictor import Predictor, get_predictor
from api.services.evaulator import Evaluator, get_evaluator