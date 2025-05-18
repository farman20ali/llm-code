import os
import sys
from flask import Flask
import tiktoken

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# from app.services.ai_service import AIService
from app.services.sql_service import SQLService

def count_tokens(text, model="gpt-4"):
    """Count tokens for a given text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback approximate method: ~4 characters per token
        return len(text) // 4

def main():
    """Compare token usage between standard and schema-aware approaches"""
    # Create a minimal Flask app context for testing
    app = Flask(__name__)
    app.logger.setLevel('INFO')
    
    # Get the path to the scripts folder (where SQL files are located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Sample query
    query = "Find all severe accidents in Karachi from last month with more than 2 casualties"
    
    # Define test cases
    tests = [
        {
            "name": "Standard Approach (with schema)",
            "config": {"USE_SCHEMA_AWARE_MODEL": False},
        },
        {
            "name": "Schema-Aware Model Approach",
            "config": {"USE_SCHEMA_AWARE_MODEL": True},
        }
    ]
    
    # Run tests
    print("\n===== Token Usage Comparison =====")
    print(f"Query: \"{query}\"\n")
    
    for test in tests:
        # Configure app
        app.config.update(test["config"])
        app.config["SCHEMA_FOLDER"] = script_dir
        
        with app.app_context():
            # Generate the messages that would be sent to the API
            if test["config"]["USE_SCHEMA_AWARE_MODEL"]:
                messages = [
                    {"role": "system", "content": (
                        "You are an assistant that generates SQL queries for a Traffic Accident Reporting System. "
                        "ONLY output a single valid SQL SELECT statement, and nothing else. "
                        "Do not include markdown formatting, code blocks, or backticks in your response."
                    )},
                    {"role": "user", "content": query}
                ]
            else:
                schema = SQLService.get_schema(script_dir)
                messages = [
                    {"role": "system", "content": (
                        "You are an assistant that ONLY outputs a single valid SQL SELECT statement, "
                        "and nothing else. Do not include markdown formatting, code blocks, or backticks in your response. "
                        "Return just the SQL query."
                    )},
                    {"role": "system", "content": f"Database schema definitions:\n{schema}"},
                    {"role": "user", "content": query}
                ]
            
            # Save messages for comparison
            test["messages"] = messages
            
            # Calculate total tokens
            total_tokens = sum(count_tokens(msg["content"]) for msg in messages)
            
            # Print results
            print(f"{test['name']}:")
            print(f"  Total tokens: {total_tokens:,}")
            
            # Break down by message
            for i, msg in enumerate(messages):
                token_count = count_tokens(msg["content"])
                print(f"  Message {i+1} ({msg['role']}): {token_count:,} tokens")
            
            print()
    
    # Calculate savings
    standard_tokens = sum(count_tokens(msg["content"]) for msg in tests[0]["messages"])
    schema_aware_tokens = sum(count_tokens(msg["content"]) for msg in tests[1]["messages"])
    
    savings = (1 - (schema_aware_tokens / standard_tokens)) * 100
    print(f"Token savings: {savings:.1f}%")
    print(f"Cost savings estimate: {savings:.1f}%")

if __name__ == "__main__":
    main() 