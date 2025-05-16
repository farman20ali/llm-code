#!/usr/bin/env python3
"""
Script to delete a fine-tuned model from OpenAI.
"""

import os
import sys
from flask import Flask
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from app.services.ai_service import AIService

def main():
    """Delete a fine-tuned model."""
    # Load environment variables
    load_dotenv()
    
    # Create a minimal Flask app context
    app = Flask(__name__)
    app.logger.setLevel('INFO')
    
    # Configure OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not found.")
        print("Please set your OpenAI API key in the .env file or environment variables.")
        return
        
    app.config['OPENAI_API_KEY'] = api_key
    
    # Get model ID from command line argument
    if len(sys.argv) != 2:
        print("Usage: python3 delete_model.py <model_id>")
        print("\nTo get a list of your models, run: python3 list_models.py")
        return
        
    model_id = sys.argv[1]
    
    with app.app_context():
        try:
            # Confirm deletion
            print(f"\nAre you sure you want to delete model {model_id}?")
            confirm = input("Type 'yes' to confirm: ")
            
            if confirm.lower() != 'yes':
                print("Deletion cancelled.")
                return
            
            # Delete the model
            success = AIService.delete_fine_tuned_model(model_id)
            
            if success:
                print(f"\nSuccessfully deleted model: {model_id}")
            else:
                print(f"\nFailed to delete model: {model_id}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
            print("\nMake sure you have:")
            print("1. A valid OpenAI API key in your .env file or environment variables")
            print("2. Sufficient permissions to access the OpenAI API")
            print("3. The correct model ID")

if __name__ == "__main__":
    main() 