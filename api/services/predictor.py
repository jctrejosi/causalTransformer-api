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
        """
        self.run_id = run_id
        self.model_type = model_type
        self.submodel = submodel
        
        if local_path:
            model, config, scaling_params = load_model_from_path(local_path, model_type, submodel)
        else:
            model, config, scaling_params = load_model_from_run(run_id, model_type, submodel)
        
        self.model = model
        self.config = config
        self.scaling_params = scaling_params  # dict con 'output_means' y 'output_stds'
        
        # Determinar si es un modelo encoder-decoder
        self.is_encoder_decoder = model_type in ["crn", "edct", "rmsn"]
        
        # Definir extractor de outcome según tipo de modelo
        if model_type == "rmsn" and submodel == "encoder":
            # RMSNEncoder retorna (outcome_pred, r)
            self.outcome_extractor = lambda out: out[0]
        elif model_type == "gnet":
            # GNet retorna directamente el tensor (vitals_outcome_pred)
            self.outcome_extractor = lambda out: out
        elif model_type == "msm":
            # MSMRegressor no tiene forward, pero usaremos su método predict
            self.outcome_extractor = None
        elif model_type in ["crn", "edct", "ct"]:
            # Estos retornan (treatment, outcome, br)
            self.outcome_extractor = lambda out: out[1]
        else:
            # Fallback genérico
            self.outcome_extractor = lambda out: out[1] if isinstance(out, tuple) else out
        
        # Determinar qué método usar para predicciones
        if hasattr(self.model, "get_predictions") and model_type != "msm":
            self.predict_method = self.model.get_predictions
        else:
            self.predict_method = self._predict_forward
    
    def _predict_forward(self, dataset) -> np.ndarray:
        """
        Método de predicción usando forward
        """
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to("cpu") for k, v in batch.items()}
                outputs = self.model(batch)
                if self.outcome_extractor is not None:
                    outcome_pred = self.outcome_extractor(outputs)
                else:
                    # Para MSM, no debería llegar aquí porque usamos predict directamente
                    outcome_pred = outputs
                predictions.append(outcome_pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def predict(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Realiza predicciones sobre nuevos datos
        """
        # Validar que los datos tienen la estructura correcta
        required_keys = ["prev_treatments", "current_treatments", "static_features"]
        if self.is_encoder_decoder:
            required_keys.append("prev_outputs")
        
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise ValueError(f"Faltan keys requeridas: {missing_keys}")
        
        # Asegurar que active_entries existe
        if 'active_entries' not in data:
            data['active_entries'] = np.ones(data['prev_treatments'].shape[:2] + (1,))
        
        # Crear dataset temporal
        import torch
        from torch.utils.data import Dataset
        
        class TemporaryDataset(Dataset):
            def __init__(self, data_dict, scaling_params=None):
                self.data = data_dict
                self.sequence_lengths = data_dict.get('sequence_lengths', 
                    np.ones(data_dict['active_entries'].shape[0]) * data_dict['active_entries'].shape[1])
                self.scaling_params = scaling_params
                self.norm_const = 1.0  # Valor por defecto, no usado en predicción
            
            def __len__(self):
                return len(self.data['active_entries'])
            
            def __getitem__(self, idx):
                return {k: torch.tensor(v[idx]) for k, v in self.data.items() if k != 'sequence_lengths'}
        
        temp_dataset = TemporaryDataset(data, scaling_params=self.scaling_params)
        
        # Realizar predicción
        if self.model_type == "msm":
            # MSM: usar método get_predictions directamente
            predictions = self.model.get_predictions(temp_dataset)
        else:
            predictions = self.predict_method(temp_dataset)
        
        # Desnormalizar si hay parámetros de escala
        if self.scaling_params is not None:
            output_means = self.scaling_params.get('output_means')
            output_stds = self.scaling_params.get('output_stds')
            if output_means is not None and output_stds is not None:
                # Asegurar que las dimensiones coinciden
                if isinstance(output_means, (list, np.ndarray)):
                    output_means = np.array(output_means)
                    output_stds = np.array(output_stds)
                    # Expandir dimensiones si es necesario (batch, time, features)
                    if predictions.ndim == 3 and output_means.ndim == 1:
                        predictions = predictions * output_stds[np.newaxis, np.newaxis, :] + output_means[np.newaxis, np.newaxis, :]
                    else:
                        predictions = predictions * output_stds + output_means
                else:
                    # Escalares
                    predictions = predictions * output_stds + output_means
        
        return predictions
    
    def predict_one_step(self, data: Dict[str, Any]) -> np.ndarray:
        """Predicción one-step-ahead"""
        return self.predict(data)
    
    def predict_n_step(self, data: Dict[str, Any], n_steps: int) -> np.ndarray:
        """Predicción multi-step"""
        # Si el modelo tiene método get_autoregressive_predictions
        if hasattr(self.model, "get_autoregressive_predictions") and self.model_type != "msm":
            from torch.utils.data import Dataset
            
            class TempAutoregDataset(Dataset):
                def __init__(self, data_dict, n_steps, scaling_params=None):
                    self.data = data_dict
                    self.n_steps = n_steps
                    self.data_processed_seq = {
                        'active_entries': np.ones((len(data_dict['active_entries']), n_steps, 1)),
                        'outputs': data_dict.get('outputs', np.zeros((len(data_dict['active_entries']), n_steps, 1))),
                        'unscaled_outputs': data_dict.get('outputs', np.zeros((len(data_dict['active_entries']), n_steps, 1)))
                    }
                    self.scaling_params = scaling_params
                    self.subset_name = "temp"
                    self.norm_const = 1.0
                
                def __len__(self):
                    return len(self.data['active_entries'])
                
                def __getitem__(self, idx):
                    return {k: torch.tensor(v[idx]) for k, v in self.data.items()}
            
            temp_dataset = TempAutoregDataset(data, n_steps, scaling_params=self.scaling_params)
            predictions = self.model.get_autoregressive_predictions(temp_dataset)
            # Desnormalizar
            if self.scaling_params is not None:
                output_means = self.scaling_params.get('output_means')
                output_stds = self.scaling_params.get('output_stds')
                if output_means is not None and output_stds is not None:
                    if predictions.ndim == 3 and isinstance(output_means, (list, np.ndarray)):
                        output_means = np.array(output_means)
                        output_stds = np.array(output_stds)
                        predictions = predictions * output_stds[np.newaxis, np.newaxis, :] + output_means[np.newaxis, np.newaxis, :]
                    else:
                        predictions = predictions * output_stds + output_means
            return predictions
        
        # Fallback: predicción iterativa
        predictions = []
        current_data = data.copy()
        for step in range(n_steps):
            step_pred = self.predict(current_data)
            predictions.append(step_pred)
            if 'prev_outputs' in current_data:
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
    """Obtiene un predictor (con caché)"""
    cache_key = f"{run_id}_{model_type}_{submodel or 'full'}"
    if use_cache and cache_key in _predictor_cache:
        return _predictor_cache[cache_key]
    
    predictor = Predictor(run_id, model_type, submodel)
    _predictor_cache[cache_key] = predictor
    return predictor