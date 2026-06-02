"""
Pydantic Models for Credit Risk API - Task 6
Pydantic V2 compatible

Defines request and response schemas for the FastAPI endpoints.

Author: Bati Bank Analytics Team
Date: June 2, 2026
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class CreditRiskRequest(BaseModel):
    """
    Request model for credit risk prediction.
    
    Features expected by the model:
    - TotalTransactionAmount: Sum of all transaction amounts
    - AverageTransactionAmount: Average transaction amount
    - TransactionCount: Number of transactions
    - StdDevTransactionAmount: Standard deviation of transaction amounts
    - Recency: Days since last transaction
    """
    
    TotalTransactionAmount: float = Field(
        ..., 
        ge=0, 
        description="Total sum of transaction amounts (must be >= 0)"
    )
    AverageTransactionAmount: float = Field(
        ..., 
        ge=0, 
        description="Average transaction amount (must be >= 0)"
    )
    TransactionCount: int = Field(
        ..., 
        ge=1, 
        description="Number of transactions (must be >= 1)"
    )
    StdDevTransactionAmount: float = Field(
        ..., 
        ge=0, 
        description="Standard deviation of transaction amounts (must be >= 0)"
    )
    Recency: int = Field(
        ..., 
        ge=0, 
        description="Days since last transaction (must be >= 0)"
    )
    
    @field_validator('TotalTransactionAmount', 'AverageTransactionAmount', 'StdDevTransactionAmount')
    @classmethod
    def validate_non_negative(cls, v: float, info) -> float:
        """Ensure values are non-negative"""
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v
    
    @field_validator('TransactionCount')
    @classmethod
    def validate_positive_int(cls, v: int, info) -> int:
        """Ensure transaction count is positive"""
        if v < 1:
            raise ValueError("TransactionCount must be at least 1")
        return v
    
    @field_validator('Recency')
    @classmethod
    def validate_recency(cls, v: int, info) -> int:
        """Ensure recency is within reasonable range"""
        if v < 0:
            raise ValueError("Recency cannot be negative")
        if v > 730:  # 2 years
            raise ValueError("Recency exceeds 2 years (730 days)")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "TotalTransactionAmount": 15000.0,
                "AverageTransactionAmount": 750.0,
                "TransactionCount": 20,
                "StdDevTransactionAmount": 500.0,
                "Recency": 15
            }
        }


class BatchCreditRiskRequest(BaseModel):
    """
    Request model for batch predictions.
    """
    customers: List[CreditRiskRequest] = Field(
        ..., 
        min_length=1, 
        max_length=100,
        description="List of customers to predict (1-100 per request)"
    )


class CreditRiskResponse(BaseModel):
    """
    Response model for credit risk prediction.
    """
    risk_probability: float = Field(
        ..., 
        ge=0, 
        le=1,
        description="Probability of being high risk (0-1)"
    )
    risk_class: str = Field(
        ..., 
        description="Risk class: 'High Risk' or 'Low Risk'"
    )
    confidence: float = Field(
        ..., 
        ge=0, 
        le=1,
        description="Model confidence in prediction (0-1)"
    )
    
    @field_validator('risk_class')
    @classmethod
    def validate_risk_class(cls, v: str) -> str:
        """Ensure risk class is valid"""
        if v not in ['High Risk', 'Low Risk']:
            raise ValueError("risk_class must be 'High Risk' or 'Low Risk'")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "risk_probability": 0.87,
                "risk_class": "High Risk",
                "confidence": 0.95
            }
        }


class BatchCreditRiskResponse(BaseModel):
    """
    Response model for batch predictions.
    """
    predictions: List[CreditRiskResponse]
    total_customers: int
    high_risk_count: int
    low_risk_count: int


class HealthResponse(BaseModel):
    """
    Health check response model.
    """
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    message: str