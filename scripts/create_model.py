#!/usr/bin/env python3
"""
Simple wrapper script to create a fine-tuned model with default settings.
Just run this script directly without arguments.
"""

import os
import sys
import subprocess

def main():
    """Run the setup_schema_model.py script with default settings"""
    # Get the path to setup_schema_model.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    setup_script = os.path.join(script_dir, 'setup_schema_model.py')
    
    # Check if the setup script exists
    if not os.path.exists(setup_script):
        print(f"Error: Setup script not found at {setup_script}")
        return 1
    
    print("Creating fine-tuned model with default settings...")
    print("This will use the full schema (no compression) for best quality.")
    print("The model will be used for all future queries once created.")
    print("\nThis process will take several minutes to generate examples,")
    print("and the actual fine-tuning will run in the background for 1-4 hours.")
    
    # Run the setup script
    result = subprocess.run([sys.executable, setup_script])
    return result.returncode

if __name__ == "__main__":
    sys.exit(main()) 