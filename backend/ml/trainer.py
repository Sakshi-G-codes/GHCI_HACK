import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class TransactionCategorizerTrainer:
    def __init__(self, config_path=None):
        if config_path is None:
            # Handle both running from backend/ and project root
            if os.path.exists("config/categories.json"):
                self.config_path = "config/categories.json"
            elif os.path.exists("../config/categories.json"):
                self.config_path = "../config/categories.json"
            else:
                self.config_path = "config/categories.json"
        else:
            self.config_path = config_path
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            stop_words='english'
        )
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.label_encoder = LabelEncoder()
        self.pipeline = None
        self.categories = None
        
    def load_categories(self):
        """Load category taxonomy from config file"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        self.categories = {cat['id']: cat for cat in config['categories']}
        return self.categories
    
    def load_training_data(self, data_path=None):
        """Load training data"""
        if data_path is None:
            # Handle both running from backend/ and project root
            if os.path.exists("data/processed/training_data.csv"):
                data_path = "data/processed/training_data.csv"
            elif os.path.exists("../data/processed/training_data.csv"):
                data_path = "../data/processed/training_data.csv"
            else:
                data_path = "data/processed/training_data.csv"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found at {data_path}. Please generate data first.")
        df = pd.read_csv(data_path)
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Clean transaction strings
        df['transaction_string'] = df['transaction_string'].str.lower().str.strip()
        
        # Remove nulls
        df = df.dropna(subset=['transaction_string', 'category'])
        
        X = df['transaction_string'].values
        y = df['category'].values
        
        return X, y
    
    def train(self, X, y):
        """Train the model"""
        print("Training transaction categorizer...")
        print(f"Training samples: {len(X)}")
        print(f"Categories: {len(set(y))}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        # Train
        self.pipeline.fit(X, y_encoded)
        
        print("Training completed!")
        return self.pipeline
    
    def save_model(self, model_path=None, encoder_path=None):
        """Save trained model"""
        if model_path is None:
            if os.path.exists("ml"):
                model_path = "ml/model.pkl"
            else:
                os.makedirs("ml", exist_ok=True)
                model_path = "ml/model.pkl"
        
        if encoder_path is None:
            if os.path.exists("ml"):
                encoder_path = "ml/label_encoder.pkl"
            else:
                os.makedirs("ml", exist_ok=True)
                encoder_path = "ml/label_encoder.pkl"
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.pipeline, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Model saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.pipeline.predict(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, confusion_matrix, classification_report
        )
        
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)
        
        # Per-class F1
        f1_per_class = f1_score(y_test_encoded, y_pred, average=None, zero_division=0)
        class_names = self.label_encoder.classes_
        f1_dict = {class_names[i]: float(f1_per_class[i]) for i in range(len(class_names))}
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        print("\n=== Evaluation Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall: {recall:.4f}")
        print(f"Macro F1-Score: {f1_macro:.4f}")
        print("\nPer-class F1-Scores:")
        for cls, f1 in f1_dict.items():
            print(f"  {cls}: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'macro_f1': f1_macro,
            'per_class_f1': f1_dict,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test_encoded, y_pred, target_names=class_names)
        }

def main():
    trainer = TransactionCategorizerTrainer()
    trainer.load_categories()
    
    # Load data
    df = trainer.load_training_data()
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    trainer.train(X_train, y_train)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    trainer.save_model()
    
    # Save evaluation metrics
    import json
    with open("ml/evaluation_metrics.json", "w") as f:
        json.dump({
            'macro_f1': metrics['macro_f1'],
            'per_class_f1': metrics['per_class_f1'],
            'confusion_matrix': metrics['confusion_matrix'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        }, f, indent=2)
    
    print("\n=== Training Complete ===")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    
    return metrics

if __name__ == "__main__":
    main()

