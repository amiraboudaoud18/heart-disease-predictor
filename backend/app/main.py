import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

env_path = Path(__file__).parent.parent.parent / ".env.development"
load_dotenv(env_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    from app.predict import _initialize

    _initialize()
    yield


app = FastAPI(
    title="Heart Disease Predictor API",
    description="Predicts heart disease risk from patient metrics",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float | None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "model_stage": os.getenv("MODEL_STAGE", "Production"),
        "model_name": os.getenv("MODEL_NAME", "heart-disease-model"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(patient: PatientData):
    try:
        from app.predict import predict

        features = patient.model_dump()
        result = predict(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("BACKEND_PORT", 8000)),
        reload=True,
    )
