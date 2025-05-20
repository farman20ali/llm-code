from app import create_app
import os

# Create the Flask application instance
app = create_app()

# Example usage in the console when running this file
if __name__ == '__main__':
    # Get the port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Print usage information
    # Print sample curl commands for each endpoint
    print("\n=== Sample API Endpoints ===")
    
    print("\n1. Natural Language to SQL:")
    print('curl -X POST http://localhost:5000/api/ask \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"question": "How many accidents occurred in the last week?"}\'')
    
    print("\n2. SQL to Insights:")
    print('curl -X POST http://localhost:5000/api/sql-insights \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"sql": "SELECT * FROM accidents LIMIT 10"}\'')
    
    print("\n3. JSON to Insights:")
    print('curl -X POST http://localhost:5000/api/json-insights \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"data": {"accidentTypeDistribution": [{"label": "Minor Collision", "count": 91}]}}\'')
    
    print("\n4. Accident Prediction:")
    print('curl -X POST http://localhost:5000/api/prediction/predict \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"area": "Downtown", "datetime": "2024-03-20T14:30:00"}\'')
    
    print("\n5. Natural Language Prediction Query:")
    print('curl -X POST http://localhost:5000/api/prediction/ask \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"question": "What is the accident risk in Downtown area tomorrow at 2 PM?"}\'')
    
    print("\n6. Generate SQL Query:")
    print('curl -X POST http://localhost:5000/api/sql/generate-sql \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"question": "Show me all accidents in the last month"}\'')
    
    print("\n7. Generate Query and Insights:")
    print('curl -X POST http://localhost:5000/query/generate-query-and-insights \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"question": "What are the most common types of accidents?"}\'')
    
    print("\n8. System Check:")
    print('curl -X GET http://localhost:5000/api/system-check')
    
    print("\n9. Get Available Tables:")
    print('curl -X GET http://localhost:5000/api/tables')
    
    print("\n10. Get Insights from Query:")
    print('curl -X POST http://localhost:5000/api/insights \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"query": "SELECT * FROM accidents LIMIT 5", "context": "Accident analysis"}\'')
    # Start the development server
    app.run(debug=True, host="0.0.0.0", port=port) 