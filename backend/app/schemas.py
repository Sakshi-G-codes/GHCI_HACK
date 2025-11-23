from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class CategorySchema(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    
    class Config:
        from_attributes = True

class TransactionCreate(BaseModel):
    transaction_string: str = Field(..., description="Raw transaction string to categorize")
    amount: Optional[float] = Field(None, description="Transaction amount")

class TransactionResponse(BaseModel):
    id: int
    transaction_string: str
    amount: Optional[float]
    predicted_category: str
    confidence_score: float
    category_id: Optional[int] = None
    
    class Config:
        from_attributes = True

class BatchCategorizeRequest(BaseModel):
    transactions: List[TransactionCreate]

class BatchCategorizeResponse(BaseModel):
    results: List[TransactionResponse]

class FeedbackRequest(BaseModel):
    transaction_id: int
    corrected_category: str
    feedback_reason: Optional[str] = None

class ExplanationResponse(BaseModel):
    transaction_id: int
    feature_attributions: Dict[str, float]
    top_keywords: List[str]
    reasoning: str

class EvaluationMetrics(BaseModel):
    macro_f1: float
    per_class_f1: Dict[str, float]
    confusion_matrix: List[List[int]]
    accuracy: float
    precision: float
    recall: float

class CategoryUpdate(BaseModel):
    categories: List[CategorySchema]

