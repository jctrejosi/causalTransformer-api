"""
Servicio para predicciones con modelos entrenados
"""
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
from omegaconf import DictConfig

from api.utils.model_loader import load_model_from_run, load_model_from_path
from api.config import RUNS_DIR

logger = logging.getLogger(__name__)


class Predictor:
    """
    Clase para realizar predicciones con modelos entrenados
    """
    
    def __init__(
        self,
        run_id: str,
        model_type: str,
        submodel: Optional[str] = None,
        local_path: Optional[Path] = None
    ):
        """
        Inicializa el predictor
        
        Args:
            run_id: ID de la run en MLflow
            model_type: Tipo de modelo
            submodel: Submodelo a usar (encoder, decoder, etc.)
            local_path: Ruta local de la run (si no se usa MLflow)
        """
        self.run_id = run_id
        self.model_type = model_type
        self.submodel = submodel
        
        if local_path:
            self.model, self.config = load_model_from_path(local_path, model_type, submodel)
        else:
            self.model, self.config = load_model_from_run(run_id, model_type, submodel)
        
        # Determinar si es un modelo encoder-decoder
        self.is_encoder_decoder = model_type in ["crn", "edct", "rmsn"]
        
        # Determinar qué método usar para predicciones
        if hasattr(self.model, "get_predictions"):
            self.predict_method = self.model.get_predictions
        else:
            self.predict_method = self._predict_forward
    
    def _predict_forward(self, dataset) -> np.ndarray:
        """
        Método de predicción fallback usando forward
        """
        # Crear DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                # Mover a CPU
                batch = {k: v.to("cpu") for k, v in batch.items()}
                # Para modelos BRCausal, forward retorna (treatment_pred, outcome_pred, br)
                outputs = self.model(batch)
                if isinstance(outputs, tuple):
                    outcome_pred = outputs[1]  # outcome_pred
                else:
                    outcome_pred = outputs
                predictions.append(outcome_pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def predict(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Realiza predicciones sobre nuevos datos
        
        Args:
            data: Diccionario con los datos (formato similar a los datasets)
            
        Returns:
            Array con las predicciones
        """
        # Validar que los datos tienen la estructura correcta
        required_keys = ["prev_treatments", "current_treatments", "static_features"]
        if self.is_encoder_decoder:
            required_keys.append("prev_outputs")
        
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise ValueError(f"Faltan keys requeridas: {missing_keys}")
        
        # Convertir a tensores y crear un dataset temporal
        import torch
        from torch.utils.data import Dataset
        
        class TemporaryDataset(Dataset):
            def __init__(self, data_dict):
                self.data = data_dict
                self.sequence_lengths = data_dict.get('sequence_lengths', 
                    np.ones(data_dict['active_entries'].shape[0]) * data_dict['active_entries'].shape[1])
                self.scaling_params = None
            
            def __len__(self):
                return len(self.data['active_entries'])
            
            def __getitem__(self, idx):
                return {k: torch.tensor(v[idx]) for k, v in self.data.items() if k != 'sequence_lengths'}
        
        # Asegurar que active_entries existe
        if 'active_entries' not in data:
            data['active_entries'] = np.ones(data['prev_treatments'].shape[:2] + (1,))
        
        # Crear dataset temporal
        temp_dataset = TemporaryDataset(data)
        
        # Realizar predicción
        predictions = self.predict_method(temp_dataset)

        # Desnormalizar si hay parámetros de escala
        if hasattr(temp_dataset, 'scaling_params') and temp_dataset.scaling_params is not None:
            output_stds = temp_dataset.scaling_params.get('output_stds')
            output_means = temp_dataset.scaling_params.get('output_means')
            if output_stds is not None and output_means is not None:
                predictions = predictions * output_stds + output_means
        
        return predictions
    
    def predict_one_step(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Predicción one-step-ahead
        """
        return self.predict(data)
    
    def predict_n_step(
        self, 
        data: Dict[str, Any], 
        n_steps: int
    ) -> np.ndarray:
        """
        Predicción multi-step
        
        Args:
            data: Datos de entrada
            n_steps: Número de pasos a predecir
        
        Returns:
            Array con predicciones de shape (batch, n_steps, output_dim)
        """
        # Si el modelo tiene método get_autoregressive_predictions
        if hasattr(self.model, "get_autoregressive_predictions"):
            # Preparar dataset para autoregresivo
            from torch.utils.data import Dataset
            
            class TempAutoregDataset(Dataset):
                def __init__(self, data_dict, n_steps):
                    self.data = data_dict
                    self.n_steps = n_steps
                    self.data_processed_seq = {
                        'active_entries': np.ones((len(data_dict['active_entries']), n_steps, 1)),
                        'outputs': data_dict.get('outputs', np.zeros((len(data_dict['active_entries']), n_steps, 1))),
                        'unscaled_outputs': data_dict.get('outputs', np.zeros((len(data_dict['active_entries']), n_steps, 1)))
                    }
                    self.scaling_params = None
                    self.subset_name = "temp"
                
                def __len__(self):
                    return len(self.data['active_entries'])
                
                def __getitem__(self, idx):
                    return {k: torch.tensor(v[idx]) for k, v in self.data.items()}
            
            temp_dataset = TempAutoregDataset(data, n_steps)
            predictions = self.model.get_autoregressive_predictions(temp_dataset)
            return predictions
        
        # Fallback: predicción iterativa
        predictions = []
        current_data = data.copy()
        
        for step in range(n_steps):
            step_pred = self.predict(current_data)
            predictions.append(step_pred)
            
            # Actualizar inputs para el siguiente paso
            if 'prev_outputs' in current_data:
                # Desplazar outputs
                current_data['prev_outputs'] = np.roll(current_data['prev_outputs'], -1, axis=1)
                current_data['prev_outputs'][:, -1:] = step_pred[:, -1:]
        
        return np.stack(predictions, axis=1)


# Cache de predictores
_predictor_cache = {}


def get_predictor(
    run_id: str,
    model_type: str,
    submodel: Optional[str] = None,
    use_cache: bool = True
) -> Predictor:
    """
    Obtiene un predictor (con caché)
    """
    cache_key = f"{run_id}_{model_type}_{submodel or 'full'}"
    
    if use_cache and cache_key in _predictor_cache:
        return _predictor_cache[cache_key]
    
    predictor = Predictor(run_id, model_type, submodel)
    _predictor_cache[cache_key] = predictor
    return predictor