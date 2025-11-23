"""
Setup script for the Financial Transaction Categorisation System
"""

import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def main():
    print("="*60)
    print("Financial Transaction Categorisation System - Setup")
    print("="*60)
    
    # Change to backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    
    # Step 1: Install dependencies
    print("\n[1/5] Installing Python dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("Warning: Some dependencies may have failed to install")
    
    # Step 2: Generate data
    print("\n[2/5] Generating synthetic transaction data...")
    if not run_command("python data/generate_data.py", "Generating data"):
        print("Error: Failed to generate data")
        return False
    
    # Step 3: Train model
    print("\n[3/5] Training ML model...")
    if not run_command("python ml/trainer.py", "Training model"):
        print("Error: Failed to train model")
        return False
    
    # Step 4: Run evaluation
    print("\n[4/5] Running evaluation...")
    if not run_command("python evaluation.py", "Evaluating model"):
        print("Warning: Evaluation may have issues, but continuing...")
    
    # Step 5: Initialize database
    print("\n[5/5] Initializing database...")
    try:
        from app.database import init_db
        init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Warning: Database initialization had issues: {e}")
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start the backend server: uvicorn app.main:app --reload --port 8000")
    print("2. Open frontend/index.html in a web browser")
    print("3. Or serve the frontend with: python -m http.server 8080")
    print("\n" + "="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

