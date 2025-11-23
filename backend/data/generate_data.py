import pandas as pd
import numpy as np
import json
import os
from typing import List, Tuple

def generate_synthetic_transactions(num_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic transaction data for training and evaluation
    
    This creates realistic transaction strings with known categories
    """
    np.random.seed(42)
    
    # Load category definitions - handle path resolution
    config_path = "config/categories.json"
    if not os.path.exists(config_path):
        config_path = "../config/categories.json"
    if not os.path.exists(config_path):
        # Try from project root
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "categories.json")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    categories = config['categories']
    
    # Define transaction patterns for each category
    transaction_patterns = {
        "dining": [
            "Starbucks Coffee", "McDonald's", "Pizza Hut", "Burger King",
            "Subway Restaurant", "Taco Bell", "KFC", "Domino's Pizza",
            "Chipotle Mexican Grill", "Panera Bread", "Dunkin Donuts",
            "Local Cafe", "Italian Restaurant", "Chinese Restaurant",
            "Sushi Bar", "Steakhouse", "Seafood Restaurant", "Food Delivery",
            "Uber Eats", "DoorDash", "Grubhub", "Restaurant", "Cafe", "Bakery"
        ],
        "shopping": [
            "Amazon.com", "Walmart", "Target", "Best Buy", "Home Depot",
            "Costco", "Macy's", "Nordstrom", "eBay", "Etsy", "Shopify",
            "Online Store", "Retail Store", "Department Store", "Fashion Store",
            "Electronics Store", "Bookstore", "Toy Store", "Gift Shop"
        ],
        "fuel": [
            "Shell Gas Station", "Exxon Mobil", "Chevron", "BP Gas",
            "Texaco", "ARCO", "7-Eleven Gas", "Speedway", "Valero",
            "Gas Station", "Fuel Pump", "Petrol Station"
        ],
        "groceries": [
            "Safeway", "Kroger", "Whole Foods", "Trader Joe's", "Aldi",
            "Costco Wholesale", "Walmart Supercenter", "Target Grocery",
            "Grocery Store", "Supermarket", "Food Market", "Farmers Market"
        ],
        "transport": [
            "Uber", "Lyft", "Taxi Service", "Metro Transit", "Subway",
            "Bus Service", "Train Station", "Airport", "Delta Airlines",
            "United Airlines", "American Airlines", "Parking Garage",
            "Parking Meter", "Toll Road"
        ],
        "utilities": [
            "Electric Company", "Water Department", "Gas Company",
            "Internet Service", "Phone Bill", "AT&T", "Verizon",
            "Comcast", "Cable TV", "Utility Bill"
        ],
        "entertainment": [
            "Netflix", "Spotify", "Movie Theater", "Cinema", "AMC",
            "Regal Cinemas", "Game Store", "Steam", "PlayStation Store",
            "Xbox Store", "Concert Ticket", "Event Ticket"
        ],
        "healthcare": [
            "CVS Pharmacy", "Walgreens", "Hospital", "Medical Clinic",
            "Doctor's Office", "Dentist", "Pharmacy", "Health Insurance",
            "Medical Lab", "Urgent Care"
        ],
        "education": [
            "University", "College", "School", "Online Course", "Coursera",
            "Udemy", "Textbook Store", "Student Services", "Tuition Payment"
        ],
        "insurance": [
            "Car Insurance", "Home Insurance", "Health Insurance",
            "Life Insurance", "Insurance Premium", "Insurance Payment"
        ],
        "banking": [
            "ATM Withdrawal", "Bank Fee", "Chase Bank", "Bank of America",
            "Wells Fargo", "Transfer Fee", "Wire Transfer", "Bank Service"
        ],
        "other": [
            "Unknown Transaction", "Misc Payment", "Other Service",
            "Uncategorized", "General Payment"
        ]
    }
    
    # Generate transactions
    transactions = []
    labels = []
    
    for category in categories:
        category_id = category['id']
        patterns = transaction_patterns.get(category_id, ["Unknown"])
        
        # Generate samples for this category
        samples_per_category = num_samples // len(categories)
        
        for _ in range(samples_per_category):
            # Select a random pattern
            base_pattern = np.random.choice(patterns)
            
            # Add some variation
            variations = [
                base_pattern,
                f"{base_pattern} #{np.random.randint(1000, 9999)}",
                f"{base_pattern} - {np.random.choice(['Store', 'Location', 'Branch'])}",
                f"{base_pattern.lower()}",
                f"{base_pattern.upper()}",
                f"  {base_pattern}  ",  # With whitespace
                f"{base_pattern.replace(' ', '-')}",
                f"{base_pattern.replace(' ', '_')}",
            ]
            
            transaction = np.random.choice(variations)
            transactions.append(transaction)
            labels.append(category_id)
    
    # Add some noise (typos, abbreviations)
    noisy_transactions = []
    noisy_labels = []
    
    for tx, label in zip(transactions[:len(transactions)//10], labels[:len(labels)//10]):
        # Add typos
        if len(tx) > 5:
            char_list = list(tx)
            idx = np.random.randint(0, len(char_list))
            char_list[idx] = np.random.choice(['x', 'z', 'q'])
            noisy_tx = ''.join(char_list)
            noisy_transactions.append(noisy_tx)
            noisy_labels.append(label)
    
    # Combine
    all_transactions = transactions + noisy_transactions
    all_labels = labels + noisy_labels
    
    # Shuffle
    indices = np.random.permutation(len(all_transactions))
    all_transactions = [all_transactions[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_string': all_transactions,
        'category': all_labels
    })
    
    return df

def main():
    print("Generating synthetic transaction data...")
    
    # Get script directory and handle paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    processed_dir = os.path.join(data_dir, "processed")
    raw_dir = os.path.join(data_dir, "raw")
    
    # Create directories
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Change to backend directory for relative paths
    backend_dir = os.path.dirname(script_dir)
    original_dir = os.getcwd()
    try:
        os.chdir(backend_dir)
        
        # Generate training data
        df_train = generate_synthetic_transactions(num_samples=5000)
        
        # Save
        train_path = os.path.join(processed_dir, "training_data.csv")
        df_train.to_csv(train_path, index=False)
        print(f"Generated {len(df_train)} training samples")
        print(f"Categories: {df_train['category'].nunique()}")
        print(f"Saved to {train_path}")
        
        # Generate test data
        df_test = generate_synthetic_transactions(num_samples=1000)
        test_path = os.path.join(processed_dir, "test_data.csv")
        df_test.to_csv(test_path, index=False)
        print(f"Generated {len(df_test)} test samples")
        print(f"Saved to {test_path}")
        
        # Print category distribution
        print("\nCategory distribution:")
        print(df_train['category'].value_counts())
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()

