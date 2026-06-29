# Causal Transformer API

API para entrenamiento y predicción de modelos causales (CRN, EDCT, CT, G-Net, RMSN, MSM).

## Endpoints

### Entrenamiento

- `POST /train/` - Inicia un entrenamiento.
- `GET /train/{job_id}/status` - Estado de un trabajo.
- `GET /train/runs` - Lista de runs.

### Predicción

- `POST /predict/` - Predicción (one-step o multi-step).
- `POST /predict/one-step` - One-step.
- `POST /predict/n-step` - Multi-step.

### Evaluación

- `POST /evaluate/` - Evalúa modelo en datos de test.
- `POST /evaluate/n-step` - Evaluación multi-step.

## Uso

Ejemplo de entrenamiento:

```bash
curl -X POST http://localhost:8000/train/ -H "Content-Type: application/json" -d '{"model_type":"ct","overrides":["dataset.num_patients.train=512"]}'
