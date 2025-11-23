import joblib
import os
import json
import numpy as np
from typing import Dict, Tuple, List

class TransactionCategorizerPredictor:
    def __init__(self, model_path=None, encoder_path=None, config_path=None):
        # Handle path resolution
        if model_path is None:
            if os.path.exists("ml/model.pkl"):
                self.model_path = "ml/model.pkl"
            elif os.path.exists("../ml/model.pkl"):
                self.model_path = "../ml/model.pkl"
            else:
                self.model_path = "ml/model.pkl"
        else:
            self.model_path = model_path
        
        if encoder_path is None:
            if os.path.exists("ml/label_encoder.pkl"):
                self.encoder_path = "ml/label_encoder.pkl"
            elif os.path.exists("../ml/label_encoder.pkl"):
                self.encoder_path = "../ml/label_encoder.pkl"
            else:
                self.encoder_path = "ml/label_encoder.pkl"
        else:
            self.encoder_path = encoder_path
        
        if config_path is None:
            if os.path.exists("config/categories.json"):
                self.config_path = "config/categories.json"
            elif os.path.exists("../config/categories.json"):
                self.config_path = "../config/categories.json"
            else:
                self.config_path = "config/categories.json"
        else:
            self.config_path = config_path
        
        self.pipeline = None
        self.label_encoder = None
        self.categories = None
        self.load_model()
        self.load_categories()
    
    def load_model(self):
        """Load trained model and label encoder"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please train the model first.")
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Label encoder not found at {self.encoder_path}. Please train the model first.")
        
        self.pipeline = joblib.load(self.model_path)
        self.label_encoder = joblib.load(self.encoder_path)
    
    def load_categories(self):
        """Load category taxonomy"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        self.categories = {cat['id']: cat for cat in config['categories']}
    
    def preprocess(self, transaction_string: str) -> str:
        """Preprocess transaction string"""
        return transaction_string.lower().strip()
    
    def predict(self, transaction_string: str) -> Tuple[str, float]:
        """
        Predict category for a transaction string
        
        Returns:
            Tuple of (category_id, confidence_score)
        """
        # Preprocess
        processed = self.preprocess(transaction_string)
        
        # Predict
        prediction_encoded = self.pipeline.predict([processed])[0]
        category_id = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence (probability)
        probabilities = self.pipeline.predict_proba([processed])[0]
        confidence = float(probabilities[prediction_encoded])
        
        return category_id, confidence
    
    def predict_batch(self, transaction_strings: List[str]) -> List[Tuple[str, float]]:
        """Predict categories for multiple transactions"""
        processed = [self.preprocess(ts) for ts in transaction_strings]
        predictions_encoded = self.pipeline.predict(processed)
        categories = self.label_encoder.inverse_transform(predictions_encoded)
        
        probabilities = self.pipeline.predict_proba(processed)
        confidences = [float(probs[pred]) for probs, pred in zip(probabilities, predictions_encoded)]
        
        return list(zip(categories, confidences))
    
    def get_top_features(self, transaction_string: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top contributing features for a prediction"""
        processed = self.preprocess(transaction_string)
        
        # Get feature names from vectorizer
        vectorizer = self.pipeline.named_steps['vectorizer']
        classifier = self.pipeline.named_steps['classifier']
        
        # Transform to TF-IDF
        tfidf = vectorizer.transform([processed])
        
        # Get feature importance for the predicted class
        prediction_encoded = self.pipeline.predict([processed])[0]
        feature_importances = classifier.feature_importances_
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get non-zero features for this transaction
        tfidf_array = tfidf.toarray()[0]
        non_zero_indices = np.where(tfidf_array > 0)[0]
        
        # Calculate weighted importance (TF-IDF * feature importance)
        weighted_importance = tfidf_array[non_zero_indices] * feature_importances[non_zero_indices]
        
        # Get top features
        top_indices = np.argsort(weighted_importance)[-top_n:][::-1]
        top_features = [(feature_names[non_zero_indices[i]], float(weighted_importance[i])) 
                       for i in top_indices]
        
        return top_features

