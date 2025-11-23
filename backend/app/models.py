from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class Category(Base):
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, index=True)
    category_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    transactions = relationship("Transaction", back_populates="category")

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_string = Column(String, nullable=False, index=True)
    amount = Column(Float)
    category_id = Column(Integer, ForeignKey("categories.id"))
    predicted_category = Column(String)
    confidence_score = Column(Float)
    is_corrected = Column(Boolean, default=False)
    user_feedback = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    category = relationship("Category", back_populates="transactions")
    explanations = relationship("Explanation", back_populates="transaction")

class Explanation(Base):
    __tablename__ = "explanations"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"))
    feature_attributions = Column(Text)  # JSON string
    top_keywords = Column(Text)  # JSON string
    reasoning = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    transaction = relationship("Transaction", back_populates="explanations")

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"))
    original_prediction = Column(String)
    corrected_category = Column(String)
    feedback_reason = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

