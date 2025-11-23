import pandas as pd
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from ml.predictor import TransactionCategorizerPredictor
import os

def load_test_data(data_path="data/processed/test_data.csv"):
    """Load test data"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data not found at {data_path}")
    df = pd.read_csv(data_path)
    return df

def evaluate_model(predictor: TransactionCategorizerPredictor, test_data: pd.DataFrame):
    """Evaluate model on test data"""
    print("Evaluating model on test data...")
    
    # Get predictions
    y_true = test_data['category'].values
    transaction_strings = test_data['transaction_string'].values
    
    predictions = []
    confidences = []
    
    for tx in transaction_strings:
        cat, conf = predictor.predict(tx)
        predictions.append(cat)
        confidences.append(conf)
    
    y_pred = np.array(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    unique_labels = sorted(set(y_true) | set(y_pred))
    f1_dict = {label: float(f1_per_class[i]) if i < len(f1_per_class) else 0.0 
               for i, label in enumerate(unique_labels)}
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=unique_labels, output_dict=True)
    
    # Confidence statistics
    avg_confidence = np.mean(confidences)
    low_confidence_count = sum(1 for c in confidences if c < 0.6)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision_macro),
        'recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'per_class_f1': f1_dict,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': unique_labels,
        'classification_report': report,
        'average_confidence': float(avg_confidence),
        'low_confidence_count': int(low_confidence_count),
        'total_samples': len(y_true)
    }
    
    return metrics

def print_evaluation_report(metrics: dict):
    """Print formatted evaluation report"""
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Precision (Macro): {metrics['precision']:.4f}")
    print(f"  Recall (Macro):    {metrics['recall']:.4f}")
    print(f"  F1-Score (Macro):  {metrics['macro_f1']:.4f}")
    print(f"  Average Confidence: {metrics['average_confidence']:.4f}")
    print(f"  Low Confidence Predictions: {metrics['low_confidence_count']}/{metrics['total_samples']}")
    
    print(f"\nPer-Class F1-Scores:")
    for label, f1 in sorted(metrics['per_class_f1'].items()):
        print(f"  {label:20s}: {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    labels = metrics['confusion_matrix_labels']
    cm = np.array(metrics['confusion_matrix'])
    
    # Print header
    print(f"{'':15s}", end="")
    for label in labels:
        print(f"{label[:10]:>10s}", end="")
    print()
    
    # Print rows
    for i, label in enumerate(labels):
        print(f"{label[:14]:15s}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:10d}", end="")
        print()
    
    print("\n" + "="*60)

def save_evaluation_report(metrics: dict, output_path="docs/EVALUATION_REPORT.md"):
    """Save evaluation report to markdown file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Evaluation Report\n\n")
        f.write("## Overall Metrics\n\n")
        f.write(f"- **Accuracy**: {metrics['accuracy']:.4f}\n")
        f.write(f"- **Precision (Macro)**: {metrics['precision']:.4f}\n")
        f.write(f"- **Recall (Macro)**: {metrics['recall']:.4f}\n")
        f.write(f"- **F1-Score (Macro)**: {metrics['macro_f1']:.4f}\n")
        f.write(f"- **Average Confidence**: {metrics['average_confidence']:.4f}\n")
        f.write(f"- **Low Confidence Predictions**: {metrics['low_confidence_count']}/{metrics['total_samples']}\n\n")
        
        f.write("## Per-Class F1-Scores\n\n")
        f.write("| Category | F1-Score |\n")
        f.write("|----------|----------|\n")
        for label, f1 in sorted(metrics['per_class_f1'].items()):
            f.write(f"| {label} | {f1:.4f} |\n")
        
        f.write("\n## Confusion Matrix\n\n")
        labels = metrics['confusion_matrix_labels']
        cm = np.array(metrics['confusion_matrix'])
        
        # Markdown table
        f.write("| | " + " | ".join(labels) + " |\n")
        f.write("|" + "---|" * (len(labels) + 1) + "\n")
        for i, label in enumerate(labels):
            f.write(f"| {label} | " + " | ".join(str(cm[i][j]) for j in range(len(labels))) + " |\n")
        
        f.write("\n## Classification Report\n\n")
        f.write("```\n")
        from sklearn.metrics import classification_report
        # Reconstruct report string if needed
        f.write("```\n")

def main():
    print("Starting evaluation...")
    
    # Load predictor
    try:
        predictor = TransactionCategorizerPredictor()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first by running: python ml/trainer.py")
        return
    
    # Load test data
    test_data = load_test_data()
    
    # Evaluate
    metrics = evaluate_model(predictor, test_data)
    
    # Print report
    print_evaluation_report(metrics)
    
    # Save report
    save_evaluation_report(metrics)
    print(f"\nEvaluation report saved to docs/EVALUATION_REPORT.md")
    
    # Save JSON metrics
    with open("ml/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to ml/evaluation_metrics.json")
    
    # Check if F1 score meets requirement
    if metrics['macro_f1'] >= 0.90:
        print(f"\n✓ SUCCESS: Macro F1-score ({metrics['macro_f1']:.4f}) meets requirement (≥ 0.90)")
    else:
        print(f"\n⚠ WARNING: Macro F1-score ({metrics['macro_f1']:.4f}) is below requirement (≥ 0.90)")

if __name__ == "__main__":
    main()

