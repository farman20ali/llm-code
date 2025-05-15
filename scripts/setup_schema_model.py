#!/usr/bin/env python3
"""
Schema Model Setup Script

This script sets up a schema-aware model for SQL generation by:
1. Loading the SQL schema from script files
2. Generating fine-tuning data for the SQL schema
3. Creating a fine-tuned model OR using an existing model ID
4. Saving the configuration for the application to use

The configuration is saved to a file that will be loaded by the application
at startup, eliminating the need to regenerate the model each time.
"""

import os
import sys
import json
import argparse
import openai
from pathlib import Path
import logging
# Add the parent directory to the path so we can import from app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ai_service import AIService
from flask import Flask

# Configuration file path
CONFIG_FILE = '.schema_model_config.json'

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Set up schema-aware model for SQL generation')
    parser.add_argument('--model-id', '-m', 
                        help='Existing fine-tuned model ID to use (skips model creation)')
    parser.add_argument('--output', '-o', 
                        default=CONFIG_FILE,
                        help=f'Output config file path (default: {CONFIG_FILE})')
    parser.add_argument('--examples', '-e', 
                        type=int, 
                        default=15,
                        help='Number of examples to generate for fine-tuning (default: 15)')
    parser.add_argument('--create-model', '-c', 
                        action='store_true',
                        default=True,
                        help='Create a new fine-tuned model (default: True)')
    parser.add_argument('--cost-tier', '-t',
                        choices=['economy', 'standard', 'premium'],
                        default='economy',
                        help='Cost tier to use for the model (default: economy)')
    parser.add_argument('--use-schema-aware', '-s',
                        action='store_true',
                        default=True,
                        help='Use schema-aware approach (default: True)')
    parser.add_argument('--compress-schema', 
                        action='store_true',
                        default=False,
                        help='Compress schema to reduce token usage (default: False)')
    return parser.parse_args()

def create_app():
    """Create a Flask app instance for AI service initialization"""
    app = Flask(__name__)
    setup_logger(app)
    # Configure the app with minimal settings
    app.config['SCHEMA_FOLDER'] = os.path.join(os.path.dirname(__file__))
    app.config['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
    return app



def setup_logger(app):
    # Enable debug level logging
    app.logger.setLevel(logging.DEBUG)

    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    handler.setFormatter(formatter)

    # Avoid duplicate logs
    if not app.logger.handlers:
        app.logger.addHandler(handler)

    # Optional: disable Werkzeug logs if noisy
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

def main():
    """Main function to set up the schema model"""
    args = parse_args()
    
    # Create a Flask app for the AI service
    app = create_app()
    
    # Initialize the AI service
    with app.app_context():
        ai_service = AIService()
        
        # Check if we need to create a model or use an existing one
        model_id = args.model_id
        
        if args.create_model or not model_id:
            print(f"Generating fine-tuning data with batch {args.examples} examples...")
            
            # # Generate fine-tuning data
            # data_file = ai_service.generate_fine_tuning_data(
            #     num_examples=args.examples,
            #     compress_schema=args.compress_schema
            # )
            # Generate fine-tuning data
            data_file = ai_service.generate_fine_tuning_data_batch(
                num_examples=args.examples,
                compress_schema=args.compress_schema
            )
            
            if not data_file:
                print("Failed to generate fine-tuning data")
                return 1
                
            print(f"Fine-tuning data generated: {data_file}")
            
            # Create the model
            print("Creating fine-tuned model (this may take a while)...")
            model_id = ai_service.create_fine_tuned_model(data_file)
            
            if not model_id:
                print("Failed to create fine-tuned model")
                return 1
                
            print(f"Fine-tuned model created: {model_id}")
        else:
            # Verify the model exists
            try:
                openai.models.retrieve(model_id)
                print(f"Using existing model: {model_id}")
            except Exception as e:
                print(f"Error retrieving model {model_id}: {e}")
                return 1
        
        # Save the configuration
        config = {
            'SQL_MODEL': model_id,
            'USE_SCHEMA_AWARE_MODEL': args.use_schema_aware,
            'COST_TIER': args.cost_tier
        }
        
        with open(args.output, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"Configuration saved to {args.output}")
        print("Schema model setup complete!")
        
        # Display usage instructions
        print("\nUsage instructions:")
        print("1. The application will now use this model automatically on startup")
        print("2. You can override settings with environment variables:")
        print("   - SQL_MODEL")
        print("   - USE_SCHEMA_AWARE_MODEL")
        print("   - COST_TIER")
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 