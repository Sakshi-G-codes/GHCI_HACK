# Dataset Documentation

## Overview

This project uses a synthetic transaction dataset generated programmatically to train and evaluate the transaction categorization model. The dataset is designed to simulate real-world financial transaction strings with known category labels.

## Dataset Generation

The dataset is generated using `backend/data/generate_data.py`, which creates realistic transaction strings based on predefined patterns for each category.

### Generation Process

1. **Category Definitions**: Categories are defined in `backend/config/categories.json` with associated keywords and patterns.

2. **Transaction Patterns**: Each category has a set of transaction patterns (e.g., "Starbucks Coffee", "Amazon.com", "Shell Gas Station").

3. **Variation**: The generator adds natural variations to transaction strings:
   - Case variations (uppercase, lowercase)
   - Whitespace variations
   - Format variations (hyphens, underscores)
   - Transaction IDs and location suffixes

4. **Noise Injection**: A small percentage of transactions include typos and abbreviations to test model robustness.

## Dataset Structure

### Training Data
- **File**: `backend/data/processed/training_data.csv`
- **Size**: ~5,000 samples
- **Format**: CSV with columns:
  - `transaction_string`: The raw transaction text
  - `category`: The ground truth category ID

### Test Data
- **File**: `backend/data/processed/test_data.csv`
- **Size**: ~1,000 samples
- **Format**: Same as training data

## Categories

The dataset includes the following categories:

1. **dining** - Dining & Restaurants
2. **shopping** - Shopping & Retail
3. **fuel** - Fuel & Gas
4. **groceries** - Groceries
5. **transport** - Transportation
6. **utilities** - Utilities
7. **entertainment** - Entertainment
8. **healthcare** - Healthcare
9. **education** - Education
10. **insurance** - Insurance
11. **banking** - Banking & Finance
12. **other** - Other/Uncategorized

## Data Distribution

The dataset is balanced across categories, with approximately equal samples per category. This ensures fair evaluation and prevents bias toward majority classes.

## Usage

### Generating the Dataset

```bash
cd backend
python data/generate_data.py
```

This will create:
- `data/processed/training_data.csv` - Training dataset
- `data/processed/test_data.csv` - Test dataset

### Using Real Data

To use your own transaction data:

1. Format your data as CSV with columns: `transaction_string`, `category`
2. Replace the files in `data/processed/`
3. Retrain the model using `python ml/trainer.py`

## Data Preprocessing

The model applies the following preprocessing steps:

1. **Lowercasing**: All transaction strings are converted to lowercase
2. **Whitespace Normalization**: Leading/trailing whitespace is removed
3. **TF-IDF Vectorization**: Text is converted to numerical features using:
   - N-gram range: (1, 3) - captures unigrams, bigrams, and trigrams
   - Max features: 5000
   - Min document frequency: 2
   - English stop words removal

## Bias Considerations

The synthetic dataset is designed to:

1. **Avoid Geographic Bias**: Transaction patterns are generic and not tied to specific regions
2. **Avoid Merchant Bias**: Multiple merchants per category prevent overfitting to specific brands
3. **Avoid Amount Bias**: Transaction amounts are not used in categorization (if provided, they're stored but not used in the model)

## Limitations

- The synthetic dataset may not capture all real-world transaction variations
- Some edge cases and domain-specific terminology may not be represented
- For production use, consider augmenting with real transaction data

## Future Improvements

- Integration with public financial datasets
- User-provided transaction data (with privacy considerations)
- Active learning from user feedback
- Domain-specific category expansion

