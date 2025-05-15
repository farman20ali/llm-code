import os
import sys
import re

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from app.services.sql_service import SQLService

def count_patterns(text, pattern):
    """Count occurrences of a pattern in text"""
    return len(re.findall(pattern, text, re.S | re.I))

def main():
    """Test the schema loader and print statistics"""
    # Get the path to the scripts folder (where SQL files are located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the schema
    print("Loading schema from:", script_dir)
    schema = SQLService.load_schema_from_folder(script_dir)
    
    # Print statistics
    total_chars = len(schema)
    create_tables = count_patterns(schema, r"CREATE TABLE")
    insert_stmts = count_patterns(schema, r"INSERT INTO")
    lov_sections = count_patterns(schema, r"-- LOV DATA:")
    
    print("\nSchema Loading Statistics:")
    print(f"Total characters loaded: {total_chars:,}")
    print(f"CREATE TABLE statements: {create_tables}")
    print(f"INSERT statements: {insert_stmts}")
    print(f"LOV DATA sections: {lov_sections}")
    
    # Print the first and last 500 characters
    print("\nFirst 500 characters:")
    print(schema[:500])
    print("\n...\n")
    print("Last 500 characters:")
    print(schema[-500:])

if __name__ == "__main__":
    main() 