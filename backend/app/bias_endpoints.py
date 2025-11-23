"""
Additional API endpoints for bias analysis and robustness testing
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from app.database import get_db
from ml.predictor import TransactionCategorizerPredictor
from ml.bias_mitigation import BiasAnalyzer, RobustnessTester

router = APIRouter(prefix="/api/bias", tags=["bias"])

class RobustnessTestRequest(BaseModel):
    transaction_string: str
    test_type: str = "all"  # "case", "whitespace", "typo", "all"

@router.post("/analyze")
async def analyze_bias(
    transactions: List[str],
    db: Session = Depends(get_db)
):
    """Analyze potential biases in a set of transactions"""
    try:
        predictor = TransactionCategorizerPredictor()
        analyzer = BiasAnalyzer(predictor)
        
        # Get predictions
        predictions = []
        confidences = []
        for tx in transactions:
            cat, conf = predictor.predict(tx)
            predictions.append(cat)
            confidences.append(conf)
        
        # Analyze biases
        merchant_bias = analyzer.analyze_merchant_bias(transactions, predictions)
        distribution = analyzer.analyze_category_distribution(predictions)
        low_conf_bias = analyzer.detect_low_confidence_bias(transactions, confidences)
        
        return {
            "merchant_bias": merchant_bias,
            "category_distribution": distribution,
            "low_confidence_patterns": low_conf_bias,
            "total_transactions": len(transactions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/robustness")
async def test_robustness(
    request: RobustnessTestRequest,
    db: Session = Depends(get_db)
):
    """Test model robustness to input variations"""
    try:
        predictor = TransactionCategorizerPredictor()
        tester = RobustnessTester(predictor)
        
        results = {}
        
        if request.test_type in ["case", "all"]:
            results["case_variations"] = tester.test_case_variations(request.transaction_string)
        
        if request.test_type in ["whitespace", "all"]:
            results["whitespace_variations"] = tester.test_whitespace_variations(request.transaction_string)
        
        if request.test_type in ["typo", "all"]:
            results["typo_robustness"] = tester.test_typo_robustness(request.transaction_string)
        
        return {
            "transaction": request.transaction_string,
            "test_type": request.test_type,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

