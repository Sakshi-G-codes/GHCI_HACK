from typing import Dict, List, Tuple
from ml.predictor import TransactionCategorizerPredictor

class TransactionExplainer:
    def __init__(self, predictor: TransactionCategorizerPredictor):
        self.predictor = predictor
    
    def explain(self, transaction_string: str) -> Dict:
        """
        Generate explanation for a prediction
        
        Returns:
            Dictionary with feature attributions, top keywords, and reasoning
        """
        # Get prediction
        category_id, confidence = self.predictor.predict(transaction_string)
        
        # Get top features
        top_features = self.predictor.get_top_features(transaction_string, top_n=15)
        
        # Build feature attributions dictionary
        feature_attributions = {feature: score for feature, score in top_features}
        
        # Extract top keywords (features with highest scores)
        top_keywords = [feature for feature, _ in top_features[:10]]
        
        # Generate reasoning
        category_name = self.predictor.categories.get(category_id, {}).get('name', category_id)
        reasoning = self._generate_reasoning(transaction_string, category_id, category_name, top_keywords, confidence)
        
        return {
            'category_id': category_id,
            'category_name': category_name,
            'confidence': confidence,
            'feature_attributions': feature_attributions,
            'top_keywords': top_keywords,
            'reasoning': reasoning
        }
    
    def _generate_reasoning(self, transaction_string: str, category_id: str, 
                           category_name: str, top_keywords: List[str], 
                           confidence: float) -> str:
        """Generate human-readable reasoning for the prediction"""
        reasoning_parts = []
        
        # Confidence level
        if confidence >= 0.8:
            conf_level = "high"
        elif confidence >= 0.6:
            conf_level = "medium"
        else:
            conf_level = "low"
        
        reasoning_parts.append(
            f"The transaction '{transaction_string}' was classified as '{category_name}' "
            f"with {conf_level} confidence ({confidence:.2%})."
        )
        
        # Keyword matching
        if top_keywords:
            reasoning_parts.append(
                f"The prediction is primarily based on the following keywords: "
                f"{', '.join(top_keywords[:5])}."
            )
        
        # Category context
        category_info = self.predictor.categories.get(category_id, {})
        if category_info.get('keywords'):
            matched_keywords = [kw for kw in category_info['keywords'] 
                              if kw.lower() in transaction_string.lower()]
            if matched_keywords:
                reasoning_parts.append(
                    f"These keywords match the category definition: {', '.join(matched_keywords)}."
                )
        
        return " ".join(reasoning_parts)

