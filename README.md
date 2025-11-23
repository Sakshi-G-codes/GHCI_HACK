# Automated AI-Based Financial Transaction Categorisation System

A standalone, high-performance transaction categorisation system that achieves business-grade accuracy and transparency while eliminating external service dependencies.

## Features

- **End-to-End Autonomous Categorisation**: No third-party API dependencies
- **High Accuracy**: Macro F1-score ≥ 0.90
- **Customisable Taxonomy**: Easy category management via JSON configuration
- **Explainability**: Feature attributions and confidence scores
- **Feedback Loop**: User correction mechanism for low-confidence predictions
- **Robustness**: Handles noisy, variable transaction strings
- **Bias Mitigation**: Ethical AI considerations built-in

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── models.py            # Database models
│   │   ├── ml_model.py          # ML model implementation
│   │   ├── schemas.py           # Pydantic schemas
│   │   └── database.py          # Database connection
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Model training pipeline
│   │   ├── predictor.py         # Inference engine
│   │   └── explainer.py         # Explainability module
│   ├── data/
│   │   ├── raw/                 # Raw transaction data
│   │   ├── processed/           # Processed data
│   │   └── generate_data.py     # Synthetic data generator
│   ├── config/
│   │   └── categories.json      # Category taxonomy configuration
│   ├── requirements.txt
│   └── evaluation.py            # Evaluation scripts
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── explainability.js
├── database/
│   └── init.sql                 # Database schema
└── docs/
    ├── DATASET.md               # Dataset documentation
    └── EVALUATION_REPORT.md     # Evaluation metrics report

```

## Quick Start

For detailed step-by-step instructions, see **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)**

### Automated Setup (Recommended)

```bash
cd backend
python setup.py
python run_server.py
```

Then open `frontend/index.html` in your browser.

### Manual Setup

1. **Install Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Generate Data**
   ```bash
   python data/generate_data.py
   ```

3. **Train Model**
   ```bash
   python ml/trainer.py
   ```

4. **Start Server**
   ```bash
   python run_server.py
   # or
   uvicorn app.main:app --reload --port 8000
   ```

5. **Open Frontend**
   - Open `frontend/index.html` in browser, OR
   - Serve with: `cd frontend && python -m http.server 8080`

### API Endpoints

- `POST /api/categorize` - Categorize a transaction
- `POST /api/categorize/batch` - Batch categorization
- `POST /api/feedback` - Submit feedback for model improvement
- `GET /api/categories` - Get all categories
- `PUT /api/categories` - Update category taxonomy
- `GET /api/evaluate` - Run evaluation and get metrics
- `GET /api/explain/{transaction_id}` - Get explainability insights

## Configuration

Edit `backend/config/categories.json` to customize the category taxonomy:

```json
{
  "categories": [
    {
      "id": "dining",
      "name": "Dining & Restaurants",
      "keywords": ["restaurant", "cafe", "food", "dining"]
    },
    {
      "id": "shopping",
      "name": "Shopping",
      "keywords": ["amazon", "store", "shop", "retail"]
    }
  ]
}
```

## Evaluation

Run evaluation to generate metrics report:

```bash
cd backend
python evaluation.py
```

This generates:
- Macro F1-score
- Per-class F1-scores
- Confusion matrix
- Detailed evaluation report

## Technologies

- **Backend**: FastAPI, SQLAlchemy, scikit-learn, transformers
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Database**: SQLite (configurable to PostgreSQL)
- **ML**: scikit-learn with TF-IDF and ensemble methods

## License

MIT License

