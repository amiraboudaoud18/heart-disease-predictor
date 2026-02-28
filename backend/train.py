import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load environment variables
env_path = Path(__file__).parent.parent / ".env.development"
load_dotenv(env_path)

# DagsHub + MLflow setup
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("heart-disease-experiment")


def get_git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def get_dvc_data_version():
    try:
        dvc_file = Path(__file__).parent.parent / "data/heart.csv.dvc"
        with open(dvc_file, "r") as f:
            for line in f:
                if "md5" in line:
                    return line.split(":")[1].strip()
    except Exception:
        return "unknown"


def train(data_path: str = None):
    if data_path is None:
        data_path = str(Path(__file__).parent.parent / "data/heart.csv")

    from app.preprocess import load_and_preprocess

    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(
        data_path
    )

    git_commit = get_git_commit()
    dvc_version = get_dvc_data_version()

    print(f"Git commit: {git_commit}")
    print(f"DVC data version: {dvc_version}")
    print(f"Training samples: {len(X_train)}")

    with mlflow.start_run():
        params = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42,
        }
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dvc_data_version", dvc_version)
        mlflow.set_tag("training_samples", len(X_train))
        mlflow.set_tag("data_path", data_path)

        os.makedirs("models", exist_ok=True)
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        with open("models/feature_names.json", "w") as f:
            json.dump(feature_names, f)

        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=os.getenv("MODEL_NAME", "heart-disease-model"),
        )

        mlflow.log_artifact("models/scaler.pkl")
        mlflow.log_artifact("models/feature_names.json")

        print("Model registered in MLflow!")

    return metrics


if __name__ == "__main__":
    default_path = str(Path(__file__).parent.parent / "data/heart.csv")
    data_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    train(data_path)
