"""
Servicio para evaluación de modelos
"""
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

from api.services.predictor import Predictor

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Clase para evaluar modelos entrenados
    """
    
    def __init__(
        self,
        run_id: str,
        model_type: str,
        submodel: Optional[str] = None,
        local_path: Optional[Path] = None
    ):
        """
        Inicializa el evaluador
        
        Args:
            run_id: ID de la run en MLflow
            model_type: Tipo de modelo
            submodel: Submodelo a usar
            local_path: Ruta local de la run
        """
        self.run_id = run_id
        self.model_type = model_type
        self.submodel = submodel
        
        # Usar el predictor para cargar el modelo
        self.predictor = Predictor(run_id, model_type, submodel, local_path)
        self.model = self.predictor.model
        self.config = self.predictor.config
    
    def _create_dataset(self, data: Dict[str, Any]):
        """Crea un dataset temporal para evaluación"""
        from torch.utils.data import Dataset
        import torch
        
        class EvalDataset(Dataset):
            def __init__(self, data_dict):
                self.data = data_dict
                # Asegurar que todos los arrays tengan las mismas dimensiones
                first_key = list(data_dict.keys())[0]
                self.length = len(data_dict[first_key])
                
                # Asegurar que exista sequence_lengths
                if 'sequence_lengths' not in data_dict:
                    self.data['sequence_lengths'] = np.ones(self.length) * data_dict['active_entries'].shape[1]
                
                # Asegurar que exista scaling_params
                self.scaling_params = None
                if 'unscaled_outputs' not in data_dict:
                    self.data['unscaled_outputs'] = data_dict.get('outputs', np.zeros((self.length, 1, 1)))
            
            def __len__(self):
                return self.length
            
            def __getitem__(self, idx):
                return {k: torch.tensor(v[idx]) for k, v in self.data.items() 
                       if k not in ['sequence_lengths', 'scaling_params', 'unscaled_outputs']}
        
        return EvalDataset(data)
    
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evalúa el modelo en datos de test
        
        Args:
            test_data: Datos con ground truth (debe contener 'outputs' y 'unscaled_outputs')
        
        Returns:
            Diccionario con métricas
        """
        # Crear dataset
        dataset = self._create_dataset(test_data)
        
        # Obtener predicciones
        predictions = self.predictor.predict(test_data)
        
        # Obtener ground truth
        outputs = test_data.get('outputs')
        if outputs is None:
            raise ValueError("Los datos de test deben contener 'outputs' (ground truth)")
        
        unscaled_outputs = test_data.get('unscaled_outputs', outputs)
        active_entries = test_data.get('active_entries', np.ones_like(outputs))
        
        # Calcular métricas
        results = {}
        
        # RMSE
        mse = ((predictions - outputs) ** 2) * active_entries
        mse_all = mse.sum() / active_entries.sum()
        rmse_all = np.sqrt(mse_all)
        
        # Obtener factor de normalización
        norm_const = getattr(dataset, 'norm_const', 1.0)
        if hasattr(self.model, 'norm_const'):
            norm_const = self.model.norm_const
        
        # Para modelos BRCausal, calcular RMSE normalizado
        if hasattr(self.model, 'get_normalised_masked_rmse'):
            # Usar el método del modelo si está disponible
            if hasattr(dataset, 'subset_name'):
                dataset.subset_name = 'test'
            # Intentar usar el método original
            try:
                if hasattr(self.model, 'get_normalised_masked_rmse'):
                    result = self.model.get_normalised_masked_rmse(dataset)
                    if isinstance(result, tuple):
                        if len(result) == 2:
                            rmse_orig, rmse_all_norm = result
                            results['rmse_orig'] = float(rmse_orig)
                            results['rmse_all'] = float(rmse_all_norm)
                        elif len(result) == 3:
                            rmse_orig, rmse_all_norm, rmse_last = result
                            results['rmse_orig'] = float(rmse_orig)
                            results['rmse_all'] = float(rmse_all_norm)
                            results['rmse_last'] = float(rmse_last)
            except Exception as e:
                logger.warning(f"No se pudo usar get_normalised_masked_rmse: {e}")
                results['rmse_all'] = float(rmse_all / norm_const)
        else:
            # RMSE básico
            results['rmse_all'] = float(rmse_all)
            
            # RMSE por paso temporal
            time_rmses = np.sqrt(mse.sum(0).sum(-1) / active_entries.sum(0).sum(-1))
            results['time_rmses'] = [float(x) for x in time_rmses]
        
        # Calcular R² si es posible
        if outputs is not None:
            ss_tot = ((outputs - outputs.mean()) ** 2 * active_entries).sum()
            ss_res = mse.sum()
            if ss_tot > 0:
                r2 = 1 - (ss_res / ss_tot)
                results['r2'] = float(r2)
        
        return results
    
    def evaluate_n_step(self, test_data: Dict[str, Any], n_steps: int = 5) -> Dict[str, float]:
        """
        Evalúa el modelo en predicción multi-step
        
        Args:
            test_data: Datos con ground truth
            n_steps: Número de pasos a predecir
        
        Returns:
            Diccionario con métricas
        """
        # Obtener predicciones multi-step
        predictions = self.predictor.predict_n_step(test_data, n_steps)
        
        # Obtener ground truth (debe tener la misma estructura que la predicción)
        outputs = test_data.get('outputs')
        if outputs is None:
            raise ValueError("Los datos de test deben contener 'outputs' (ground truth)")
        
        # Si outputs no tiene la dimensión temporal adecuada
        if len(outputs.shape) == 2:  # (batch, features)
            outputs = outputs[:, np.newaxis, :]
        
        # Asegurar que predictions y outputs tienen la misma dimensión
        if outputs.shape[1] < predictions.shape[1]:
            outputs = outputs[:, :predictions.shape[1], :]
        elif outputs.shape[1] > predictions.shape[1]:
            predictions = np.pad(predictions, ((0, 0), (0, outputs.shape[1] - predictions.shape[1]), (0, 0)))
        
        # Calcular RMSE por paso
        active_entries = test_data.get('active_entries', np.ones_like(outputs))
        if len(active_entries.shape) == 2:
            active_entries = active_entries[:, :, np.newaxis]
        
        n_step_rmses = {}
        for step in range(min(predictions.shape[1], outputs.shape[1])):
            mse = ((predictions[:, step, :] - outputs[:, step, :]) ** 2) * active_entries[:, step, :]
            rmse = np.sqrt(mse.sum() / active_entries[:, step, :].sum())
            n_step_rmses[f'step_{step+1}'] = float(rmse)
        
        # Promedio de RMSE
        avg_rmse = np.mean(list(n_step_rmses.values()))
        
        return {
            'n_step_rmses': n_step_rmses,
            'avg_rmse': float(avg_rmse),
            'metrics': {'avg_rmse': float(avg_rmse)}
        }


# Cache de evaluadores
_evaluator_cache = {}


def get_evaluator(
    run_id: str,
    model_type: str,
    submodel: Optional[str] = None,
    use_cache: bool = True
) -> Evaluator:
    """
    Obtiene un evaluador (con caché)
    """
    cache_key = f"{run_id}_{model_type}_{submodel or 'full'}"
    
    if use_cache and cache_key in _evaluator_cache:
        return _evaluator_cache[cache_key]
    
    evaluator = Evaluator(run_id, model_type, submodel)
    _evaluator_cache[cache_key] = evaluator
    return evaluator