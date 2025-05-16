#!/usr/bin/env python3
"""
Script to list all fine-tuned models in the OpenAI account.
"""

import os
import sys
from flask import Flask
from datetime import datetime
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from app.services.ai_service import AIService

def format_tokens(tokens):
    """Format token count with K/M suffix."""
    if tokens >= 1_000_000:
        return f"{tokens/1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens/1_000:.1f}K"
    return str(tokens)

def main():
    """List all fine-tuned models and their details."""
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
    
    with app.app_context():
        try:
            # Get list of models
            models = AIService.list_fine_tuned_models()
            
            if not models:
                print("No fine-tuned models found in your account.")
                return
                
            print(f"\nFound {len(models)} fine-tuned models:\n")
            
            # Print model details
            for i, model in enumerate(models, 1):
                print(f"Model {i}:")
                print(f"  ID: {model['model_id']}")
                print(f"  Status: {model['status']}")
                print(f"  Base Model: {model['base_model']}")
                if model['suffix']:
                    print(f"  Suffix: {model['suffix']}")
                if model['created_at']:
                    created = datetime.fromtimestamp(model['created_at'])
                    print(f"  Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
                if model['finished_at']:
                    finished = datetime.fromtimestamp(model['finished_at'])
                    print(f"  Finished: {finished.strftime('%Y-%m-%d %H:%M:%S')}")
                if model['trained_tokens']:
                    print(f"  Trained Tokens: {format_tokens(model['trained_tokens'])}")
                if model['training_file']:
                    print(f"  Training File: {model['training_file']}")
                print()
                
        except Exception as e:
            print(f"Error: {str(e)}")
            print("\nMake sure you have:")
            print("1. A valid OpenAI API key in your .env file or environment variables")
            print("2. Sufficient permissions to access the OpenAI API")
            print("3. An active OpenAI account with fine-tuning capabilities")

if __name__ == "__main__":
    main() 