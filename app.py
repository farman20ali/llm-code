from app import create_app
import os

# Create the Flask application instance
app = create_app()

# Example usage in the console when running this file
if __name__ == '__main__':
    # Get the port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Print usage information
    print("\n=== AI SQL and Data Insights API ===")
    print("\nAvailable endpoints:")
    print("  POST /api/ask             - Natural language to SQL with insights")
    print("  POST /api/sql-insights    - SQL to insights")
    print("  POST /api/json-insights   - JSON to insights")
    print("\nExample requests:")
    print('\n  curl -X POST http://localhost:5000/api/ask -H "Content-Type: application/json" -d \'{"question": "How many accidents in the last week?"}\'')
    print('\n  curl -X POST http://localhost:5000/api/sql-insights -H "Content-Type: application/json" -d \'{"sql": "SELECT * FROM accidents LIMIT 10"}\'')
    print('\n  curl -X POST http://localhost:5000/api/json-insights -H "Content-Type: application/json" -d \'{"data": {"accidentTypeDistribution": [{"label": "Minor Collision", "count": 91}]}}\'')
    print("\nStarting development server...")
    
    # Start the development server
    app.run(debug=True, host="0.0.0.0", port=port) 