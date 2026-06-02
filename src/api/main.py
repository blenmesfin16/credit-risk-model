"""
FastAPI Credit Risk Prediction Service - Task 6

Provides REST API endpoints for credit risk prediction using the trained model.

Author: Bati Bank Analytics Team
Date: June 2, 2026
"""

import os
import sys
import logging
from typing import Optional
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.pydantic_models import (
    CreditRiskRequest,
    BatchCreditRiskRequest,
    CreditRiskResponse,
    BatchCreditRiskResponse,
    HealthResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk probability for customers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and feature columns
model = None
feature_columns = None
model_version = None


# ============================================
# MODEL LOADING
# ============================================

def load_model():
    """
    Load the best trained model from local file system.
    
    Priority:
    1. models/best_model_production.pkl (from training)
    2. models/best_tuned_model.pkl (fallback)
    3. models/best_model.pkl (fallback)
    """
    global model, feature_columns, model_version
    
    model_paths = [
        'models/best_model_production.pkl',
        'models/best_tuned_model.pkl',
        'models/best_model.pkl'
    ]
    
    feature_paths = [
        'models/feature_columns.pkl',
        'models/feature_columns.pkl'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                logger.info(f"✅ Model loaded from: {path}")
                model_version = path
                break
            except Exception as e:
                logger.error(f"Failed to load model from {path}: {e}")
    
    if model is None:
        logger.error("No model found! Run src/train.py first to train a model.")
        return False
    
    # Load feature columns
    for path in feature_paths:
        if os.path.exists(path):
            try:
                feature_columns = joblib.load(path)
                logger.info(f"✅ Feature columns loaded from: {path}")
                break
            except Exception as e:
                logger.error(f"Failed to load features from {path}: {e}")
    
    if feature_columns is None:
        # Define default feature columns
        feature_columns = [
            'TotalTransactionAmount', 'AverageTransactionAmount',
            'TransactionCount', 'StdDevTransactionAmount', 'Recency'
        ]
        logger.info("Using default feature columns")
    
    return True


def predict_single(customer_data: CreditRiskRequest) -> tuple:
    """
    Make a single prediction.
    
    Returns:
        (risk_probability, confidence)
    """
    global model, feature_columns
    
    # Convert request to DataFrame
    df = pd.DataFrame([[
        customer_data.TotalTransactionAmount,
        customer_data.AverageTransactionAmount,
        customer_data.TransactionCount,
        customer_data.StdDevTransactionAmount,
        customer_data.Recency
    ]], columns=feature_columns)
    
    # Make prediction
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(df)[0]
        risk_probability = proba[1]  # Probability of class 1 (high risk)
        confidence = max(proba)
    else:
        # Fallback for models without predict_proba
        risk_probability = float(model.predict(df)[0])
        confidence = abs(risk_probability - 0.5) * 2 + 0.5
    
    return risk_probability, confidence


# ============================================
# API ENDPOINTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    success = load_model()
    if not success:
        logger.warning("API starting without model - predictions will fail")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version=model_version,
        message="Credit Risk Prediction API is running. Use /docs for documentation."
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=model_version,
        message="Model loaded and ready" if model is not None else "Model not loaded - train first"
    )


@app.post("/predict", response_model=CreditRiskResponse)
async def predict(request: CreditRiskRequest):
    """
    Predict credit risk for a single customer.
    
    Returns risk probability (0-1) where:
    - 0 = Low Risk
    - 1 = High Risk
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first by running src/train.py"
        )
    
    try:
        risk_probability, confidence = predict_single(request)
        
        risk_class = "High Risk" if risk_probability >= 0.5 else "Low Risk"
        
        return CreditRiskResponse(
            risk_probability=round(float(risk_probability), 4),
            risk_class=risk_class,
            confidence=round(float(confidence), 4)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchCreditRiskResponse)
async def predict_batch(request: BatchCreditRiskRequest):
    """
    Predict credit risk for multiple customers in batch.
    
    Maximum 100 customers per request.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        predictions = []
        high_risk_count = 0
        low_risk_count = 0
        
        for customer in request.customers:
            risk_probability, confidence = predict_single(customer)
            risk_class = "High Risk" if risk_probability >= 0.5 else "Low Risk"
            
            if risk_class == "High Risk":
                high_risk_count += 1
            else:
                low_risk_count += 1
            
            predictions.append(CreditRiskResponse(
                risk_probability=round(float(risk_probability), 4),
                risk_class=risk_class,
                confidence=round(float(confidence), 4)
            ))
        
        return BatchCreditRiskResponse(
            predictions=predictions,
            total_customers=len(request.customers),
            high_risk_count=high_risk_count,
            low_risk_count=low_risk_count
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ============================================
# RUN THE APP
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )