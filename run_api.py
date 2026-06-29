"""
Script para probar la API
"""
import requests
import json
import time
import numpy as np

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health:", response.json())

def test_train():
    payload = {
        "model_type": "ct",
        "overrides": [
            "dataset.num_patients.train=64",
            "dataset.num_patients.val=10",
            "exp.max_epochs=2",
            "exp.gpus=0"
        ]
    }
    response = requests.post(f"{BASE_URL}/train/", json=payload)
    print("Train response:", response.json())
    return response.json().get("job_id")

def test_status(job_id):
    response = requests.get(f"{BASE_URL}/train/{job_id}/status")
    print("Status:", response.json())

def test_predict(job_id):
    # Datos de ejemplo para predicción (ajustar según tu modelo)
    data = {
        "prev_treatments": [[[0.0, 0.0]]],
        "current_treatments": [[[1.0, 0.0]]],
        "static_features": [[0.5, 1.2]],
        "prev_outputs": [[[20.0]]],
        "active_entries": [[[1.0]]]
    }
    payload = {
        "run_id": job_id,
        "model_type": "ct",
        "data": data,
        "n_steps": 3
    }
    response = requests.post(f"{BASE_URL}/predict/", json=payload)
    print("Predict:", response.json())

if __name__ == "__main__":
    test_health()
    job_id = test_train()
    time.sleep(5)
    test_status(job_id)
    # Si el entrenamiento ya terminó, probar predicción
    # test_predict(job_id)