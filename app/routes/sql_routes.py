from flask import Blueprint, request, jsonify, current_app
from app.services.langchain_service import LangChainService

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