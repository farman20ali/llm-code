import os
import sys
import time
from flask import Flask
import tiktoken

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from app.services.ai_service import AIService

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

def estimate_cost(input_tokens, output_tokens, model):
    """Estimate the cost of a request based on tokens and model"""
    # Prices per 1K tokens (as of May 2025)
    pricing = {
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        'gpt-4o-mini': {'input': 0.0015, 'output': 0.0060},
        'gpt-4o': {'input': 0.0030, 'output': 0.0060}
    }
    
    # Default to gpt-4o pricing if model not found
    model_prices = pricing.get(model, pricing['gpt-4o'])
    
    # Calculate cost
    input_cost = (input_tokens / 1000) * model_prices['input']
    output_cost = (output_tokens / 1000) * model_prices['output']
    
    return input_cost + output_cost

def main():
    """Compare token usage and costs across different model tiers"""
    # Create a minimal Flask app context for testing
    app = Flask(__name__)
    app.logger.setLevel('INFO')
    
    # Get the path to the scripts folder (where SQL files are located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test configuration for the different cost tiers
    queries = [
        "Find severe accidents in Karachi from the last month",
        "How many accidents involved motorcycles in Lahore?",
        "List all accidents with more than 2 casualties requiring hospital transfer"
    ]
    
    cost_tiers = ['economy', 'standard', 'premium']
    
    # Configure app with our model options
    app.config.update({
        "SCHEMA_FOLDER": script_dir,
        "ECONOMY_MODEL": "gpt-3.5-turbo",
        "STANDARD_MODEL": "gpt-4o-mini",
        "PREMIUM_MODEL": "gpt-4o",
        "USE_SCHEMA_AWARE_MODEL": False  # Include schema in all requests for testing
    })
    
    print("\n===== Model Cost Comparison =====\n")
    print("Using schema-inclusive approach for all models\n")
    
    # Store results for comparison
    results = {}
    
    for tier in cost_tiers:
        # Set the cost tier for this test
        app.config["COST_TIER"] = tier
        
        # Get the model for this tier
        with app.app_context():
            model = AIService.get_model_by_cost_tier(tier)
        
        print(f"--- {tier.upper()} TIER (Model: {model}) ---")
        total_input_tokens = 0
        
        # Use schema size as the main input cost since it dominates
        with app.app_context():
            schema = AIService.get_schema(script_dir)
            schema_tokens = count_tokens(schema, model)
            total_input_tokens = schema_tokens
        
        # Add query tokens (using an average query length)
        avg_query_tokens = sum(count_tokens(q, model) for q in queries) // len(queries)
        total_input_tokens += avg_query_tokens
        
        # Estimate output tokens (SQL queries are relatively short)
        estimated_output_tokens = 50  # Typical SQL query length
        
        # Calculate cost per query
        cost_per_query = estimate_cost(total_input_tokens, estimated_output_tokens, model)
        
        # Print results
        print(f"  Schema tokens: {schema_tokens:,}")
        print(f"  Average query tokens: {avg_query_tokens}")
        print(f"  Estimated output tokens: {estimated_output_tokens}")
        print(f"  Estimated cost per query: ${cost_per_query:.4f}")
        
        # For volume estimates
        print(f"  Cost for 1,000 queries: ${cost_per_query * 1000:.2f}")
        print(f"  Cost for 10,000 queries: ${cost_per_query * 10000:.2f}")
        print()
        
        # Store results
        results[tier] = {
            'model': model,
            'input_tokens': total_input_tokens,
            'output_tokens': estimated_output_tokens,
            'cost_per_query': cost_per_query
        }
    
    # Compare schema-aware vs schema-inclusive (using economy tier as example)
    print("--- SCHEMA-AWARE vs SCHEMA-INCLUSIVE COMPARISON ---")
    print("Using economy tier (gpt-3.5-turbo) as example:\n")
    
    # Schema-inclusive already calculated in results['economy']
    schema_inclusive = results['economy']
    
    # Calculate schema-aware approach (no schema tokens sent)
    with app.app_context():
        model = AIService.get_model_by_cost_tier('economy')
        system_prompt = "You are an assistant that generates SQL queries for a Traffic Accident Reporting System."
        system_tokens = count_tokens(system_prompt, model)
        query_tokens = avg_query_tokens  # Reuse from above
        
        schema_aware_input_tokens = system_tokens + query_tokens
        schema_aware_cost = estimate_cost(schema_aware_input_tokens, estimated_output_tokens, model)
    
    # Print comparison
    print(f"Schema-Inclusive Approach:")
    print(f"  Input tokens: {schema_inclusive['input_tokens']:,}")
    print(f"  Cost per query: ${schema_inclusive['cost_per_query']:.4f}")
    print(f"  Cost for 1,000 queries: ${schema_inclusive['cost_per_query'] * 1000:.2f}")
    print()
    
    print(f"Schema-Aware Approach:")
    print(f"  Input tokens: {schema_aware_input_tokens:,}")
    print(f"  Cost per query: ${schema_aware_cost:.4f}")
    print(f"  Cost for 1,000 queries: ${schema_aware_cost * 1000:.2f}")
    print()
    
    # Calculate savings
    savings_percent = (1 - (schema_aware_cost / schema_inclusive['cost_per_query'])) * 100
    savings_1k = (schema_inclusive['cost_per_query'] - schema_aware_cost) * 1000
    
    print(f"Cost savings with schema-aware approach: {savings_percent:.1f}%")
    print(f"Savings for 1,000 queries: ${savings_1k:.2f}")
    print(f"Recommendation: Use {results['economy']['model']} with schema-aware approach for best cost-efficiency")

if __name__ == "__main__":
    main() 