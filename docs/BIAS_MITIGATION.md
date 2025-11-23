# Bias Mitigation and Responsible AI

## Overview

This document outlines the bias mitigation strategies and responsible AI practices implemented in the Financial Transaction Categorisation System.

## Bias Mitigation Strategies

### 1. Balanced Training Data

- **Equal Category Representation**: The synthetic dataset ensures approximately equal samples per category
- **Stratified Splitting**: Train/test splits maintain category proportions
- **Class Weighting**: Random Forest uses `class_weight='balanced'` to handle any remaining imbalance

### 2. Feature Engineering

- **TF-IDF Vectorization**: Reduces bias toward frequent terms by normalizing term frequencies
- **N-gram Range (1-3)**: Captures context without overfitting to specific patterns
- **Stop Word Removal**: Prevents common words from dominating predictions

### 3. No Amount-Based Bias

- Transaction amounts are **not used** in categorization decisions
- Amounts are stored for record-keeping but excluded from model features
- Prevents discrimination based on transaction value

### 4. Geographic Neutrality

- Transaction patterns are generic and not tied to specific regions
- No location-based features in the model
- Merchant names are normalized to prevent regional bias

### 5. Merchant Diversity

- Multiple merchants per category prevent overfitting to specific brands
- Synthetic data includes variations to avoid merchant-specific bias
- Model generalizes across different merchant naming conventions

## Bias Detection Tools

### Bias Analyzer

The `BiasAnalyzer` class provides tools to detect potential biases:

1. **Merchant Bias Analysis**: Identifies if certain merchants are consistently misclassified
2. **Category Distribution Analysis**: Checks if predictions are balanced across categories
3. **Low Confidence Pattern Detection**: Finds transaction patterns with consistently low confidence

### Robustness Testing

The `RobustnessTester` class evaluates model robustness:

1. **Case Variation Testing**: Ensures predictions are consistent across case variations
2. **Whitespace Robustness**: Tests handling of whitespace variations
3. **Typo Tolerance**: Evaluates model's ability to handle common typos

## API Endpoints

### `/api/bias/analyze`
Analyzes potential biases in a set of transactions.

**Request:**
```json
{
  "transactions": ["Starbucks Coffee", "Amazon.com", ...]
}
```

**Response:**
```json
{
  "merchant_bias": {...},
  "category_distribution": {...},
  "low_confidence_patterns": {...}
}
```

### `/api/bias/robustness`
Tests model robustness to input variations.

**Request:**
```json
{
  "transaction_string": "Starbucks Coffee",
  "test_type": "all"
}
```

## Ethical Considerations

### Privacy

- No personally identifiable information (PII) is stored
- Transaction strings are treated as generic text patterns
- User feedback is anonymized

### Fairness

- Model treats all transaction types equally
- No discrimination based on merchant, amount, or other attributes
- Low-confidence predictions are flagged for human review

### Transparency

- Explainability features show why predictions were made
- Confidence scores indicate prediction certainty
- Users can provide feedback to improve the model

### Accountability

- All predictions are logged with confidence scores
- User feedback is tracked for model improvement
- Evaluation metrics are publicly available

## Continuous Improvement

### Feedback Loop

1. **Low Confidence Flagging**: Predictions with confidence < 0.6 are flagged
2. **User Corrections**: Users can submit corrections via the feedback API
3. **Model Retraining**: Feedback data can be used to retrain the model

### Monitoring

- Regular evaluation on test sets
- Bias analysis on production predictions
- Robustness testing on edge cases

## Limitations and Future Work

### Current Limitations

- English-only transaction strings
- Synthetic data may not capture all real-world variations
- Limited to predefined categories

### Future Improvements

- Multi-language support
- Real-world data integration (with privacy safeguards)
- Active learning from user feedback
- Advanced bias detection algorithms
- Fairness metrics beyond F1-score

## References

- [Fairness in Machine Learning](https://developers.google.com/machine-learning/fairness-overview)
- [Responsible AI Practices](https://ai.google/principles/)
- [Bias Detection in NLP](https://arxiv.org/abs/1903.10561)

