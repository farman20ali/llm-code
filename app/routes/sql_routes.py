from flask import Blueprint, request, jsonify, current_app
from app.services.langchain_service import LangChainService
from app.services.sql_service import SQLService

# Create blueprint
sql_bp = Blueprint('sql', __name__)

@sql_bp.route('/generate-sql', methods=['POST'])
def generate_sql():
    """Generate SQL query from natural language question."""
    try:
        print("/generate-sql")
        # Get question from request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Missing question in request body',
                'status': 'error'
            }), 400
            
        question = data['question']
        
        # Generate SQL
        result = LangChainService.generate_sql(question)
        
        # Return the result directly since it's already formatted
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'question': data.get('question', '')
        }), 400
    except Exception as e:
        current_app.logger.error(f"Error generating SQL: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error',
            'question': data.get('question', '')
        }), 500

@sql_bp.route('/batch-query', methods=['POST'])
def batch_query():
    """Execute multiple SQL queries and return their execution status.
    
    Example POST body (using query_list):
    {
        "query_list": [
            "SELECT * FROM accident_reports WHERE severity > 3",
            "SELECT COUNT(*) FROM accident_reports WHERE weather_condition = 1"
        ]
    }
    
    Example POST body (using queryMap):
    {
        "queryMap": {
            "severe_accidents": "SELECT * FROM accident_reports WHERE severity > 3",
            "weather_count": "SELECT COUNT(*) FROM accident_reports WHERE weather_condition = 1"
        }
    }
    
    Returns:
    {
        "successfulQueries": [
            {
                "key": "severe_accidents",  # Only present if using queryMap
                "query": "SELECT * FROM accident_reports WHERE severity > 3",
                "columns": [...],
                "rows": [...]
            }
        ],
        "unsuccessfulQueries": [
            {
                "key": "weather_count",  # Only present if using queryMap
                "query": "SELECT COUNT(*) FROM accident_reports WHERE weather_condition = 1",
                "error": "Error message here"
            }
        ]
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    # Handle both query_list and queryMap formats
    if 'query_list' in data and data['query_list'] and len(data['query_list']) > 0:
        query_list = data['query_list']
        if not isinstance(query_list, list):
            return jsonify({'error': 'query_list must be an array'}), 400
        # Convert list to map with numeric keys
        query_map = {str(i): query for i, query in enumerate(query_list)}
    elif 'queryMap' in data:
        query_map = data['queryMap']
        if not isinstance(query_map, dict):
            return jsonify({'error': 'queryMap must be an object'}), 400
    else:
        return jsonify({'error': 'Either query_list or queryMap is required'}), 400
    
    current_app.logger.info(f"Processing batch query with {len(query_map)} queries")
    
    successful_queries = []
    unsuccessful_queries = []
    
    # Get a connection from the pool
    conn = None
    try:
        conn = SQLService.get_connection()
        with conn.cursor() as cur:
            for key, query in query_map.items():
                try:
                    # Validate SQL query
                    if not SQLService.validate_select(query):
                        unsuccessful_queries.append({
                            "key": key,
                            "query": query,
                            "error": "Invalid SQL query. Only SELECT statements are allowed."
                        })
                        continue
                    
                    # Execute SQL using the same cursor
                    cur.execute(query)
                    cols = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    
                    successful_queries.append({
                        "key": key,
                        "query": query,
                        "columns": cols,
                        "rows": rows
                    })
                    
                except Exception as e:
                    unsuccessful_queries.append({
                        "key": key,
                        "query": query,
                        "error": str(e)
                    })
    except Exception as e:
        current_app.logger.error(f"Error in batch query execution: {str(e)}")
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    finally:
        # Release connection back to pool
        if conn is not None:
            SQLService.release_connection(conn)
    
    return jsonify({
        "successfulQueries": successful_queries,
        "unsuccessfulQueries": unsuccessful_queries
    }) 