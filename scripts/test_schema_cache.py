import os
import sys
import time
from flask import Flask

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from app.services.sql_service import SQLService

def main():
    """Test the schema caching functionality"""
    # Create a minimal Flask app context for logging
    app = Flask(__name__)
    app.logger.setLevel('INFO')
    
    # Get the path to the scripts folder (where SQL files are located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # First load - should read from disk
    print("\n=== First schema load (cold) ===")
    start_time = time.time()
    with app.app_context():
        schema1 = SQLService.get_schema(script_dir)
    load_time1 = time.time() - start_time
    schema_size = len(schema1)
    
    # Second load - should use cache
    print("\n=== Second schema load (cached) ===")
    start_time = time.time()
    with app.app_context():
        schema2 = SQLService.get_schema(script_dir)
    load_time2 = max(time.time() - start_time, 0.0001)  # Ensure minimum time to prevent division by zero
    
    # Print results
    print(f"\nSchema size: {schema_size:,} characters")
    print(f"First load time: {load_time1:.4f} seconds")
    print(f"Second load time: {load_time2:.4f} seconds")
    print(f"Speed improvement: {load_time1 / load_time2:.1f}x faster")
    
    if id(schema1) == id(schema2):
        print("Cache is working correctly - same object returned")
    else:
        print("Warning: Different objects returned, cache may not be working")

if __name__ == "__main__":
    main() 