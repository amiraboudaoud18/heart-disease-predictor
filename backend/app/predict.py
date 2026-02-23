import json
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env.development"
load_dotenv(env_path)

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

_model = None
_scaler = None
_feature_names = None


def _initialize():
    global _model, _scaler, _feature_names

    if _model is not None:
        return

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    print("Loading model from MLflow...")
    model_name = os.getenv("MODEL_NAME", "heart-disease-model")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    model_uri = f"models:/{model_name}/{model_stage}"
    _model = mlflow.sklearn.load_model(model_uri)
    print("Model loaded!")

    # Get run_id to download artifacts
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[model_stage])
    run_id = versions[0].run_id

    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="scaler.pkl"
    )
    with open(scaler_path, "rb") as f:
        _scaler = pickle.load(f)
    print("Scaler loaded!")

    features_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="feature_names.json"
    )
    with open(features_path, "r") as f:
        _feature_names = json.load(f)
    print("All loaded!")


def predict(features: dict) -> dict:
    _initialize()

    input_array = np.array([[features[f] for f in _feature_names]])
    input_scaled = _scaler.transform(input_array)

    prediction = _model.predict(input_scaled)
    probability = _model.predict_proba(input_scaled)[0][1]

    result = int(prediction[0])
    return {
        "prediction": result,
        "label": "High Risk" if result == 1 else "Low Risk",
        "probability": round(float(probability), 4),
    }
