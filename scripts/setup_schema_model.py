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
import time
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
                        default=True,
                        help='Compress schema to reduce token usage (default: False)')
    return parser.parse_args()

def create_app():
    """Create a Flask app instance for AI service initialization"""
    app = Flask(__name__)
    setup_logger(app)
    # Configure the app with minimal settings
    app.config['SCHEMA_FOLDER'] = os.path.join(os.path.dirname(__file__))
    app.config['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
    if not app.config['OPENAI_API_KEY']:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # Model configuration
    app.config['COST_TIER'] = os.environ.get('COST_TIER', 'economy')
    app.config['USE_SCHEMA_AWARE_MODEL'] = os.environ.get('USE_SCHEMA_AWARE_MODEL', 'true').lower() == 'true'
    
    # Model names based on cost tier
    app.config['ECONOMY_MODEL'] = os.environ.get('ECONOMY_MODEL', 'gpt-3.5-turbo')
    app.config['STANDARD_MODEL'] = os.environ.get('STANDARD_MODEL', 'gpt-4o-mini')
    app.config['PREMIUM_MODEL'] = os.environ.get('PREMIUM_MODEL', 'gpt-4o')
    
    # SQL model configuration
    app.config['SQL_MODEL'] = os.environ.get('SQL_MODEL')
    
    # Log configuration
    app.logger.info("Application configuration loaded:")
    app.logger.info(f"Cost Tier: {app.config['COST_TIER']}")
    app.logger.info(f"Use Schema Aware Model: {app.config['USE_SCHEMA_AWARE_MODEL']}")
    app.logger.info(f"Economy Model: {app.config['ECONOMY_MODEL']}")
    app.logger.info(f"Standard Model: {app.config['STANDARD_MODEL']}")
    app.logger.info(f"Premium Model: {app.config['PREMIUM_MODEL']}")
    app.logger.info(f"SQL Model: {app.config['SQL_MODEL']}")
    
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

def find_existing_training_files(schema_folder):
    """Find existing .jsonl training files in the schema folder."""
    training_files = []
    for file in os.listdir(schema_folder):
        if file.endswith('.jsonl'):
            full_path = os.path.join(schema_folder, file)
            # Get file creation time and size
            stats = os.stat(full_path)
            training_files.append({
                'path': full_path,
                'name': file,
                'created': stats.st_ctime,
                'size': stats.st_size
            })
    return sorted(training_files, key=lambda x: x['created'], reverse=True)

def prompt_for_training_file(training_files):
    """Prompt user to select a training file or generate new one."""
    if not training_files:
        print("\nNo existing training files found.")
        while True:
            choice = input("Would you like to generate a new training file? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return None
            elif choice in ['n', 'no']:
                return "STOP"
            print("Please enter 'y' or 'n'")
        
    print("\nExisting training files:")
    for i, file in enumerate(training_files, 1):
        created = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file['created']))
        size_mb = file['size'] / (1024 * 1024)
        print(f"{i}. {file['name']} (Created: {created}, Size: {size_mb:.2f} MB)")
    
    print("\nOptions:")
    print("0. Generate new training file")
    
    while True:
        try:
            choice = input("\nSelect a file number (0 for new file): ").strip()
            if choice == '0':
                while True:
                    confirm = input("Are you sure you want to generate a new training file? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        return None
                    elif confirm in ['n', 'no']:
                        return "STOP"
                    print("Please enter 'y' or 'n'")
            choice = int(choice)
            if 1 <= choice <= len(training_files):
                return training_files[choice - 1]['path']
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    """Main function to set up the schema model"""
    args = parse_args()
    
    # Create a Flask app for the AI service
    app = create_app()
    
    # Initialize the AI service
    with app.app_context():
        ai_service = AIService()
        
        # Check if we need to create a model or use an existing one
        model_id = args.model_id or os.environ.get('SQL_MODEL')
        
        if args.create_model or not model_id:
            # Look for existing training files
            training_files = find_existing_training_files(app.config['SCHEMA_FOLDER'])
            selected_file = prompt_for_training_file(training_files)
            
            if selected_file == "STOP":
                app.logger.info("Operation cancelled by user")
                return 0
            elif selected_file:
                app.logger.info(f"Using existing training file: {selected_file}")
                data_file = selected_file
            else:
                app.logger.info(f"Generating new fine-tuning data with {args.examples} examples...")
                
                # Generate fine-tuning data
                data_file = ai_service.generate_fine_tuning_data_batch(
                    num_examples=args.examples,
                    compress_schema=args.compress_schema
                )
                
                if not data_file:
                    app.logger.error("Failed to generate fine-tuning data")
                    return 1
                    
                app.logger.info(f"Fine-tuning data generated: {data_file}")
                
                # Prompt user to continue with model creation
                while True:
                    choice = input("\nTraining file has been created. Would you like to continue with model creation? (y/n): ").strip().lower()
                    if choice in ['y', 'yes']:
                        break
                    elif choice in ['n', 'no']:
                        app.logger.info("Operation stopped by user. Training file is saved for later use.")
                        return 0
                    print("Please enter 'y' or 'n'")
            
            # Create the model
            app.logger.info("Creating fine-tuned model (this may take a while)...")
            model_id = ai_service.create_fine_tuned_model(data_file)
            
            if not model_id:
                app.logger.error("Failed to create fine-tuned model")
                return 1
                
            app.logger.info(f"Fine-tuned model created: {model_id}")
        else:
            # Verify the model exists
            try:
                openai.models.retrieve(model_id)
                app.logger.info(f"Using existing model: {model_id}")
            except Exception as e:
                app.logger.error(f"Error retrieving model {model_id}: {e}")
                return 1
        
        # Save the configuration
        config = {
            'model_id': model_id,
            'use_schema_aware_model': args.use_schema_aware,
            'cost_tier': args.cost_tier,
            'temperature': 0.0,
            'max_tokens': 2048
        }
        
        with open(args.output, 'w') as f:
            json.dump(config, f, indent=2)
            
        app.logger.info(f"Configuration saved to {args.output}")
        app.logger.info("Schema model setup complete!")
        
        # Display usage instructions
        app.logger.info("\nUsage instructions:")
        app.logger.info("1. The application will now use this model automatically on startup")
        app.logger.info("2. You can override settings with environment variables:")
        app.logger.info("   - SQL_MODEL (maps to model_id)")
        app.logger.info("   - USE_SCHEMA_AWARE_MODEL (maps to use_schema_aware_model)")
        app.logger.info("   - COST_TIER (maps to cost_tier)")
        app.logger.info("   - ECONOMY_MODEL (default: gpt-3.5-turbo)")
        app.logger.info("   - STANDARD_MODEL (default: gpt-4o-mini)")
        app.logger.info("   - PREMIUM_MODEL (default: gpt-4o)")
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 