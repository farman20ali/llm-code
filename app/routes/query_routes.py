from flask import Blueprint, request, jsonify, current_app
from app.services.ai_service import AIService
from app.services.sql_service import SQLService

query_bp = Blueprint('query', __name__)

@query_bp.route('/generate-query-and-insights', methods=['POST'])
def generate_query_and_insights():
    """Generate SQL query and insights from natural language question."""
    try:
        print('/generate-query-and-insights')
        # Get question from request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Missing question in request',
                'status': 'error'
            }), 400
            
        question = data['question']
        
        # Generate SQL using LangChain
        sql_result = LangChainService.generate_sql_only(question)
        if sql_result['status'] == 'error':
            return jsonify(sql_result), 400
            
        sql = sql_result['sql']
        
        # Validate SQL query
        if not SQLService.validate_select(sql):
            return jsonify({
                'error': 'Invalid SQL query. Only SELECT statements are allowed.',
                'status': 'error',
                'sql': sql
            }), 400
        
        try:
            # Execute SQL using existing method
            cols, rows = SQLService.run_sql(sql)
            
            if not rows:
                return jsonify({
                    'error': 'No results found',
                    'status': 'error',
                    'sql': sql
                }), 404
            
            # Generate insights using AIService
            insights = AIService.generate_insight_from_sql_results(sql, cols, rows)
            
            return jsonify({
                'sql': sql,
                'question': question,
                'status': 'success',
                'results': {
                    'columns': cols,
                    'rows': rows,
                    'row_count': len(rows)
                },
                'insights': insights
            })
            
        except Exception as e:
            current_app.logger.error(f"Error executing SQL: {str(e)}")
            return jsonify({
                'error': f'Error executing SQL: {str(e)}',
                'status': 'error',
                'sql': sql
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Error in generate-query-and-insights: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@query_bp.route('/batch-generate-and-compare', methods=['POST'])
def batch_generate_and_compare():
    """Generate SQL for multiple questions and compare with expected SQL."""
    try:
        # Get questions data from request
        data = request.get_json()
        if not data or not isinstance(data, list):
            return jsonify({
                'error': 'Request must contain a list of questions with expected SQL',
                'status': 'error'
            }), 400
            
        # Validate each item has required fields
        for item in data:
            if not isinstance(item, dict) or 'question' not in item or 'expectedSql' not in item:
                return jsonify({
                    'error': 'Each item must have "question" and "expectedSql" fields',
                    'status': 'error'
                }), 400
        
        # Process batch using AIService
        results, stats = AIService.generate_and_compare_sql_batch(
            questions_data=data,
            schema_folder=current_app.config.get('SCHEMA_FOLDER')
        )
        
        return jsonify({
            'status': 'success',
            'results': results,
            'statistics': stats
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in batch-generate-and-compare: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500 