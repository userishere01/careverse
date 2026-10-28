from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import numpy as np

app = FastAPI(title="CareVerse AI Engine", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "unit4_ml", "model.pkl")
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None and os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
    return pipeline

class SymptomRequest(BaseModel):
    symptoms: str

@app.get("/")
def health_check():
    pipe = get_pipeline()
    return {"status": "active", "model_loaded": pipe is not None}

@app.post("/predict-specialist")
def predict_specialist(req: SymptomRequest):
    pipe = get_pipeline()
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model file not found. Run train.py first.")

    user_text = req.symptoms.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty symptom text.")

    probabilities = pipe.predict_proba([user_text])[0]
    classes = pipe.classes_

    top_indices = np.argsort(probabilities)[::-1][:3]
    ranked = []
    for idx in top_indices:
        ranked.append({
            "specialist": str(classes[idx]),
            "confidence_score": round(float(probabilities[idx]), 4)
        })

    critical_terms = ["chest tightness", "chest pain", "breathing difficulty", "unconscious", "stroke", "bleeding"]
    is_emergency = any(kw in user_text.lower() for kw in critical_terms)

    return {
        "input_text": user_text,
        "is_emergency_flag": is_emergency,
        "primary_recommendation": ranked[0]["specialist"],
        "confidence": ranked[0]["confidence_score"],
        "ranked_candidates": ranked
    }