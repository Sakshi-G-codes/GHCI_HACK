from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import json
import os

from app.database import get_db, init_db
from app import models, schemas
from app.bias_endpoints import router as bias_router
from ml.predictor import TransactionCategorizerPredictor
from ml.explainer import TransactionExplainer

# Initialize FastAPI app
app = FastAPI(
    title="Financial Transaction Categorisation API",
    description="AI-based transaction categorization system",
    version="1.0.0"
)

# Include bias analysis routes
app.include_router(bias_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML components
predictor = None
explainer = None

@app.on_event("startup")
async def startup_event():
    """Initialize database and ML models on startup"""
    global predictor, explainer
    
    # Initialize database
    init_db()
    
    # Load ML model
    try:
        predictor = TransactionCategorizerPredictor()
        explainer = TransactionExplainer(predictor)
        print("ML model loaded successfully")
    except FileNotFoundError as e:
        print(f"Warning: ML model not found. Please train the model first. {e}")
        predictor = None
        explainer = None
    
    # Load categories into database
    load_categories_to_db()

def load_categories_to_db():
    """Load categories from config file into database"""
    db = next(get_db())
    try:
        config_path = "config/categories.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            for cat_data in config['categories']:
                # Check if category exists
                existing = db.query(models.Category).filter(
                    models.Category.category_id == cat_data['id']
                ).first()
                
                if not existing:
                    category = models.Category(
                        category_id=cat_data['id'],
                        name=cat_data['name'],
                        description=cat_data.get('description', '')
                    )
                    db.add(category)
            
            db.commit()
            print("Categories loaded into database")
    except Exception as e:
        print(f"Error loading categories: {e}")
        db.rollback()
    finally:
        db.close()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Financial Transaction Categorisation API",
        "version": "1.0.0",
        "status": "operational" if predictor else "model_not_loaded"
    }

@app.post("/api/categorize", response_model=schemas.TransactionResponse)
async def categorize_transaction(
    transaction: schemas.TransactionCreate,
    db: Session = Depends(get_db)
):
    """Categorize a single transaction"""
    if not predictor:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Please train the model first."
        )
    
    # Predict
    category_id, confidence = predictor.predict(transaction.transaction_string)
    
    # Get category from database
    category = db.query(models.Category).filter(
        models.Category.category_id == category_id
    ).first()
    
    # Save transaction
    db_transaction = models.Transaction(
        transaction_string=transaction.transaction_string,
        amount=transaction.amount,
        predicted_category=category_id,
        confidence_score=confidence
    )
    if category:
        db_transaction.category_id = category.id
    
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    
    return schemas.TransactionResponse(
        id=db_transaction.id,
        transaction_string=db_transaction.transaction_string,
        amount=db_transaction.amount,
        predicted_category=db_transaction.predicted_category,
        confidence_score=db_transaction.confidence_score,
        category_id=db_transaction.category_id
    )

@app.post("/api/categorize/batch", response_model=schemas.BatchCategorizeResponse)
async def categorize_batch(
    request: schemas.BatchCategorizeRequest,
    db: Session = Depends(get_db)
):
    """Categorize multiple transactions in batch"""
    if not predictor:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Please train the model first."
        )
    
    results = []
    
    for tx in request.transactions:
        category_id, confidence = predictor.predict(tx.transaction_string)
        
        # Get category from database
        category = db.query(models.Category).filter(
            models.Category.category_id == category_id
        ).first()
        
        # Save transaction
        db_transaction = models.Transaction(
            transaction_string=tx.transaction_string,
            amount=tx.amount,
            predicted_category=category_id,
            confidence_score=confidence
        )
        if category:
            db_transaction.category_id = category.id
        
        db.add(db_transaction)
        db.commit()
        db.refresh(db_transaction)
        
        results.append(schemas.TransactionResponse(
            id=db_transaction.id,
            transaction_string=db_transaction.transaction_string,
            amount=db_transaction.amount,
            predicted_category=db_transaction.predicted_category,
            confidence_score=db_transaction.confidence_score,
            category_id=db_transaction.category_id
        ))
    
    return schemas.BatchCategorizeResponse(results=results)

@app.post("/api/feedback")
async def submit_feedback(
    feedback: schemas.FeedbackRequest,
    db: Session = Depends(get_db)
):
    """Submit feedback for model improvement"""
    # Get transaction
    transaction = db.query(models.Transaction).filter(
        models.Transaction.id == feedback.transaction_id
    ).first()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    # Update transaction
    transaction.is_corrected = True
    transaction.user_feedback = feedback.feedback_reason
    
    # Get corrected category
    corrected_category = db.query(models.Category).filter(
        models.Category.category_id == feedback.corrected_category
    ).first()
    
    if corrected_category:
        transaction.category_id = corrected_category.id
    
    # Save feedback
    feedback_record = models.Feedback(
        transaction_id=feedback.transaction_id,
        original_prediction=transaction.predicted_category,
        corrected_category=feedback.corrected_category,
        feedback_reason=feedback.feedback_reason
    )
    
    db.add(feedback_record)
    db.commit()
    
    return {"message": "Feedback submitted successfully", "feedback_id": feedback_record.id}

@app.get("/api/categories", response_model=List[schemas.CategorySchema])
async def get_categories(db: Session = Depends(get_db)):
    """Get all categories"""
    categories = db.query(models.Category).all()
    return [
        schemas.CategorySchema(
            id=cat.category_id,
            name=cat.name,
            description=cat.description
        )
        for cat in categories
    ]

@app.put("/api/categories")
async def update_categories(
    category_update: schemas.CategoryUpdate,
    db: Session = Depends(get_db)
):
    """Update category taxonomy"""
    # Update config file
    config_path = "config/categories.json"
    config = {
        "categories": [
            {
                "id": cat.id,
                "name": cat.name,
                "description": cat.description or ""
            }
            for cat in category_update.categories
        ]
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Update database
    for cat_data in category_update.categories:
        existing = db.query(models.Category).filter(
            models.Category.category_id == cat_data.id
        ).first()
        
        if existing:
            existing.name = cat_data.name
            existing.description = cat_data.description
        else:
            new_cat = models.Category(
                category_id=cat_data.id,
                name=cat_data.name,
                description=cat_data.description
            )
            db.add(new_cat)
    
    db.commit()
    
    # Reload model if it exists
    global predictor, explainer
    try:
        predictor = TransactionCategorizerPredictor()
        explainer = TransactionExplainer(predictor)
    except:
        pass
    
    return {"message": "Categories updated successfully"}

@app.get("/api/explain/{transaction_id}", response_model=schemas.ExplanationResponse)
async def explain_prediction(
    transaction_id: int,
    db: Session = Depends(get_db)
):
    """Get explainability insights for a prediction"""
    if not explainer:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Please train the model first."
        )
    
    # Get transaction
    transaction = db.query(models.Transaction).filter(
        models.Transaction.id == transaction_id
    ).first()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    # Generate explanation
    explanation = explainer.explain(transaction.transaction_string)
    
    # Save explanation to database
    db_explanation = models.Explanation(
        transaction_id=transaction_id,
        feature_attributions=json.dumps(explanation['feature_attributions']),
        top_keywords=json.dumps(explanation['top_keywords']),
        reasoning=explanation['reasoning']
    )
    
    db.add(db_explanation)
    db.commit()
    
    return schemas.ExplanationResponse(
        transaction_id=transaction_id,
        feature_attributions=explanation['feature_attributions'],
        top_keywords=explanation['top_keywords'],
        reasoning=explanation['reasoning']
    )

@app.get("/api/evaluate", response_model=schemas.EvaluationMetrics)
async def get_evaluation_metrics():
    """Get evaluation metrics"""
    metrics_path = "ml/evaluation_metrics.json"
    
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics not found. Please run evaluation first."
        )
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return schemas.EvaluationMetrics(
        macro_f1=metrics['macro_f1'],
        per_class_f1=metrics['per_class_f1'],
        confusion_matrix=metrics['confusion_matrix'],
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall']
    )

@app.get("/api/transactions")
async def get_transactions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get recent transactions"""
    transactions = db.query(models.Transaction).offset(skip).limit(limit).all()
    return [
        {
            "id": tx.id,
            "transaction_string": tx.transaction_string,
            "amount": tx.amount,
            "predicted_category": tx.predicted_category,
            "confidence_score": tx.confidence_score,
            "is_corrected": tx.is_corrected,
            "created_at": tx.created_at.isoformat() if tx.created_at else None
        }
        for tx in transactions
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

