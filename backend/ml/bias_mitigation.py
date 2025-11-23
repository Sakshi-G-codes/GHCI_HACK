"""
Bias Mitigation Module

This module provides utilities for detecting and mitigating bias in transaction categorization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter

class BiasAnalyzer:
    """Analyze potential biases in predictions"""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def analyze_merchant_bias(self, transactions: List[str], predictions: List[str]) -> Dict:
        """
        Analyze if certain merchants are consistently misclassified
        
        Returns:
            Dictionary with bias metrics
        """
        merchant_patterns = {}
        
        for tx, pred in zip(transactions, predictions):
            # Extract merchant name (first word or common patterns)
            merchant = self._extract_merchant(tx)
            if merchant not in merchant_patterns:
                merchant_patterns[merchant] = []
            merchant_patterns[merchant].append(pred)
        
        # Calculate consistency
        bias_report = {}
        for merchant, preds in merchant_patterns.items():
            if len(preds) > 1:
                most_common = Counter(preds).most_common(1)[0]
                consistency = most_common[1] / len(preds)
                bias_report[merchant] = {
                    'consistency': consistency,
                    'predictions': dict(Counter(preds))
                }
        
        return bias_report
    
    def analyze_category_distribution(self, predictions: List[str]) -> Dict:
        """Analyze if predictions are balanced across categories"""
        category_counts = Counter(predictions)
        total = len(predictions)
        
        distribution = {
            cat: count / total 
            for cat, count in category_counts.items()
        }
        
        # Calculate entropy (higher = more balanced)
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in distribution.values())
        max_entropy = np.log2(len(distribution))
        balance_score = entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            'distribution': distribution,
            'balance_score': balance_score,
            'entropy': entropy
        }
    
    def detect_low_confidence_bias(self, transactions: List[str], 
                                   confidences: List[float],
                                   threshold: float = 0.6) -> Dict:
        """Detect if certain transaction patterns have consistently low confidence"""
        low_confidence_patterns = {}
        
        for tx, conf in zip(transactions, confidences):
            if conf < threshold:
                pattern = self._extract_pattern(tx)
                if pattern not in low_confidence_patterns:
                    low_confidence_patterns[pattern] = []
                low_confidence_patterns[pattern].append(conf)
        
        return {
            pattern: {
                'count': len(confs),
                'avg_confidence': np.mean(confs),
                'min_confidence': np.min(confs)
            }
            for pattern, confs in low_confidence_patterns.items()
        }
    
    def _extract_merchant(self, transaction: str) -> str:
        """Extract merchant name from transaction string"""
        # Simple heuristic: first significant word
        words = transaction.lower().split()
        # Remove common prefixes
        skip_words = {'the', 'a', 'an', 'at', 'on', 'in'}
        for word in words:
            if word not in skip_words and len(word) > 2:
                return word
        return words[0] if words else "unknown"
    
    def _extract_pattern(self, transaction: str) -> str:
        """Extract pattern from transaction (for bias analysis)"""
        # Normalize transaction
        normalized = transaction.lower().strip()
        # Remove numbers
        normalized = ''.join(c for c in normalized if not c.isdigit())
        # Take first few words
        words = normalized.split()[:3]
        return ' '.join(words)

class RobustnessTester:
    """Test model robustness to noise and variations"""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def test_case_variations(self, transaction: str) -> Dict:
        """Test how predictions change with case variations"""
        variations = {
            'original': transaction,
            'lowercase': transaction.lower(),
            'uppercase': transaction.upper(),
            'title_case': transaction.title(),
            'mixed_case': self._random_case(transaction)
        }
        
        results = {}
        for variant_name, variant_text in variations.items():
            try:
                category, confidence = self.predictor.predict(variant_text)
                results[variant_name] = {
                    'category': category,
                    'confidence': confidence
                }
            except:
                results[variant_name] = {'error': 'Prediction failed'}
        
        # Check consistency
        categories = [r.get('category') for r in results.values() if 'category' in r]
        consistency = len(set(categories)) == 1 if categories else False
        
        return {
            'results': results,
            'consistent': consistency,
            'unique_categories': list(set(categories))
        }
    
    def test_whitespace_variations(self, transaction: str) -> Dict:
        """Test how predictions change with whitespace variations"""
        variations = {
            'original': transaction,
            'leading_space': '  ' + transaction,
            'trailing_space': transaction + '  ',
            'extra_spaces': '  '.join(transaction.split()),
            'no_spaces': transaction.replace(' ', '')
        }
        
        results = {}
        for variant_name, variant_text in variations.items():
            try:
                category, confidence = self.predictor.predict(variant_text)
                results[variant_name] = {
                    'category': category,
                    'confidence': confidence
                }
            except:
                results[variant_name] = {'error': 'Prediction failed'}
        
        categories = [r.get('category') for r in results.values() if 'category' in r]
        consistency = len(set(categories)) == 1 if categories else False
        
        return {
            'results': results,
            'consistent': consistency,
            'unique_categories': list(set(categories))
        }
    
    def test_typo_robustness(self, transaction: str, num_typos: int = 1) -> Dict:
        """Test how predictions change with typos"""
        import random
        
        results = {}
        for i in range(5):  # Test 5 random typo variations
            typo_transaction = self._add_typos(transaction, num_typos)
            try:
                category, confidence = self.predictor.predict(typo_transaction)
                results[f'typo_variant_{i+1}'] = {
                    'transaction': typo_transaction,
                    'category': category,
                    'confidence': confidence
                }
            except:
                results[f'typo_variant_{i+1}'] = {'error': 'Prediction failed'}
        
        # Check if original category is maintained
        original_category, _ = self.predictor.predict(transaction)
        maintained = all(
            r.get('category') == original_category 
            for r in results.values() 
            if 'category' in r
        )
        
        return {
            'original_category': original_category,
            'results': results,
            'category_maintained': maintained
        }
    
    def _random_case(self, text: str) -> str:
        """Randomly change case of characters"""
        return ''.join(
            c.upper() if np.random.random() > 0.5 else c.lower()
            for c in text
        )
    
    def _add_typos(self, text: str, num_typos: int) -> str:
        """Add random typos to text"""
        import random
        chars = list(text)
        for _ in range(num_typos):
            if len(chars) > 0:
                idx = random.randint(0, len(chars) - 1)
                # Random character substitution
                chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
        return ''.join(chars)

