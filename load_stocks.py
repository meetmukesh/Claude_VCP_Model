"""
Utility script to load stock universe from CSV into database
"""

import pandas as pd
import sys
from pathlib import Path

from database import DatabaseManager

def load_stocks_from_csv(csv_path: str):
    """Load stocks from CSV file into database"""
    
    # Initialize database
    db = DatabaseManager()
    db.create_tables()
    
    print(f"Loading stocks from {csv_path}...")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} stocks in CSV")
        
        # Load into database
        count = db.load_stock_universe(df=df)
        
        print(f"✅ Successfully loaded {count} new stocks into database")
        print(f"Total active stocks: {len(db.get_active_stocks())}")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_path}")
        print("Please ensure the CSV file exists")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python load_stocks.py <path_to_csv>")
        print("Example: python load_stocks.py stocks_L.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    load_stocks_from_csv(csv_path)

if __name__ == "__main__":
    main()
