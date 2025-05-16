#!/usr/bin/env python3
"""
Script to test schema compression and display the results.
"""

import os
import sys
from flask import Flask
import json

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from app.services.ai_service import AIService
from app.services.sql_service import SQLService

def main():
    """Test schema compression and display results."""
    # Create a minimal Flask app context
    app = Flask(__name__)
    app.logger.setLevel('INFO')
    
    # Get the path to the scripts folder (where SQL files are located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configure Flask app
    app.config.update({
        "SCHEMA_FOLDER": script_dir
    })
    
    with app.app_context():
        # Load original schema
        original_schema = SQLService.load_schema_from_folder(script_dir)
        print("\n=== Original Schema Size ===")
        print(f"Characters: {len(original_schema)}")
        print(f"Estimated tokens: ~{len(original_schema) // 4}")
        
        # Get compressed schema
        compressed_schema = AIService._compress_schema(original_schema)
        print("\n=== Compressed Schema Size ===")
        print(f"Characters: {len(compressed_schema)}")
        print(f"Estimated tokens: ~{len(compressed_schema) // 4}")
        
        # Calculate compression ratio
        compression_ratio = (1 - len(compressed_schema) / len(original_schema)) * 100
        print(f"\nCompression ratio: {compression_ratio:.1f}%")
        
        # Display compressed schema
        print("\n=== Compressed Schema Content ===")
        print(compressed_schema)

if __name__ == "__main__":
    main() 