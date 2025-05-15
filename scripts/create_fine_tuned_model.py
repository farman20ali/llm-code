#!/usr/bin/env python3
"""
Script to create a fine-tuned model for SQL generation.

This is the ultimate cost optimization for production use.
Fine-tuning a model on our schema allows us to avoid sending
the entire schema with every request, saving ~99% on token costs.
"""

import os
import sys
import argparse
from flask import Flask
import time

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from app.services.ai_service import AIService

def main():
    """Create a fine-tuned model for SQL generation"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a fine-tuned model for SQL generation')
    parser.add_argument('--output', type=str, default='training_data.jsonl',
                        help='Path to save training data (default: training_data.jsonl)')
    parser.add_argument('--examples', type=int, default=50,
                        help='Number of examples to generate (default: 50)')
    parser.add_argument('--create-model', action='store_true',
                        help='Create the fine-tuned model after generating training data')
    parser.add_argument('--model-suffix', type=str, default='accident-sql-generator',
                        help='Suffix for the fine-tuned model name (default: accident-sql-generator)')
    args = parser.parse_args()
    
    # Create a minimal Flask app context for the AIService
    app = Flask(__name__)
    app.logger.setLevel('INFO')
    
    # Get the path to the scripts folder (where SQL files are located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, args.output)
    
    # Configure Flask app
    app.config.update({
        "SCHEMA_FOLDER": script_dir,
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY")
    })
    
    print(f"\n===== Creating Fine-Tuned Model =====\n")
    print(f"Step 1: Generating {args.examples} training examples\n")
    
    start_time = time.time()
    
    # Generate training data
    with app.app_context():
        success = AIService.generate_fine_tuning_data(
            schema_folder=script_dir,
            output_file=output_file,
            num_examples=args.examples
        )
    
    if not success:
        print(f"Error: Failed to generate training data")
        return 1
    
    generation_time = time.time() - start_time
    print(f"\nTraining data generated in {generation_time:.2f} seconds")
    print(f"Training data saved to: {output_file}")
    
    # Create the fine-tuned model if requested
    if args.create_model:
        if not os.environ.get("OPENAI_API_KEY"):
            print(f"Error: OPENAI_API_KEY environment variable is not set")
            return 1
        
        print(f"\nStep 2: Creating fine-tuned model with suffix '{args.model_suffix}'\n")
        
        with app.app_context():
            try:
                job_id = AIService.create_fine_tuned_model(
                    training_file_path=output_file,
                    suffix=args.model_suffix
                )
                print(f"\nFine-tuning job created: {job_id}")
                print(f"The fine-tuning process may take several hours.")
                print(f"Once complete, set these environment variables to use your model:")
                print(f"  export USE_SCHEMA_AWARE_MODEL=true")
                print(f"  export SQL_MODEL=ft:{job_id}")
                print(f"  export COST_TIER=economy")
            except Exception as e:
                print(f"Error creating fine-tuned model: {e}")
                return 1
    else:
        print(f"\nTo create a fine-tuned model later, run:")
        print(f"  python {sys.argv[0]} --create-model --output {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 