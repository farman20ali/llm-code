from flask import Blueprint, request, jsonify, current_app
from app.services.ai_service import AIService
from app.services.sql_service import SQLService
import psycopg2
import os

bp = Blueprint('api', __name__, url_prefix='/api')

@bp.route('/ask', methods=['POST'])
def ask_database():
    print("ask_database")
    """Endpoint for natural language questions to database with AI insights.
    
    Example POST body:
    {
        "question": "How many car accidents were reported in the last week?"
    }
    """
    data = request.json
    
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400
    
    question = data['question']
    schema_folder = current_app.config['SCHEMA_FOLDER']
    
    try:
        # First test API connection
        connection_ok, connection_msg = AIService.check_api_connection()
        if not connection_ok:
            current_app.logger.error(f"API connection test failed: {connection_msg}")
            return jsonify({'error': f"OpenAI API connection error: {connection_msg}"}), 503
            
        # Generate SQL first to check validity
        try:
            # Generate the SQL query
            sql = AIService.generate_sql_from_question(question, schema_folder)
            
            # Validate SQL query
            if not SQLService.validate_select(sql):
                error_msg = f"Invalid SQL generated. The query does not appear to be a valid SELECT statement: {sql}"
                current_app.logger.error(f"SQL validation failed: {sql}")
                return jsonify({'error': error_msg}), 400
                
            # If SQL is valid, proceed with full insight generation
            result = AIService.ask_database_with_insight(question, schema_folder)
            return jsonify(result)
            
        except ValueError as e:
            # This captures SQL validation errors from ask_database_with_insight
            error_msg = str(e)
            if "Invalid SQL generated" in error_msg:
                # Extract just the SQL query from the error message if possible
                sql_start = error_msg.find(":") + 1 if ":" in error_msg else 0
                sql_query = error_msg[sql_start:].strip()
                current_app.logger.error(f"Validation error: {error_msg}")
                return jsonify({
                    'error': 'The generated SQL query is invalid',
                    'sql': sql_query,
                    'suggestion': 'Try rephrasing your question to be more specific about the data you need.'
                }), 400
            else:
                raise
                
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        current_app.logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@bp.route('/sql-insights', methods=['POST'])
def sql_insights():
    print("sql_insights")
    """Endpoint for generating insights from an SQL query.
    
    Example POST body:
    {
        "sql": "SELECT * FROM accidents WHERE date > '2023-01-01' LIMIT 10"
    }
    """
    data = request.json
    
    if not data or 'sql' not in data:
        return jsonify({'error': 'SQL query is required'}), 400
    
    sql = data['sql']
    
    # Validate SQL query
    if not SQLService.validate_select(sql):
        return jsonify({'error': 'Invalid SQL query. Only SELECT statements are allowed.'}), 400
    
    try:
        # Execute SQL
        cols, rows = SQLService.run_sql(sql)
        
        # Generate insights
        insight = AIService.generate_insight_from_sql_results(sql, cols, rows)
        
        return jsonify({
            'sql': sql,
            'columns': cols,
            'rows': rows,
            'insight': insight
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/json-insights', methods=['POST'])
def json_insights():
    print("json_insights")
    """Endpoint for generating insights from structured JSON data.
    
    Example POST body:
    {
        "data": {
            "accidentTypeDistribution": [...],
            "vehicleTypeDistribution": [...],
            ...
        }
    }
    """
    request_data = request.json
    
    if not request_data or 'data' not in request_data:
        return jsonify({'error': 'JSON data is required'}), 400
    
    try:
        # First test API connection
        connection_ok, connection_msg = AIService.check_api_connection()
        if not connection_ok:
            current_app.logger.error(f"API connection test failed: {connection_msg}")
            return jsonify({'error': f"OpenAI API connection error: {connection_msg}"}), 503
            
        # Generate insights from JSON data
        insight = AIService.generate_insights_from_json(request_data['data'])
        
        # Check if the insight is an error message
        if insight.startswith("Error generating insights:"):
            return jsonify({'error': insight}), 500
            
        return jsonify({
            'data': request_data['data'],
            'insight': insight
        })
    except Exception as e:
        error_msg = f"Failed to generate insights: {str(e)}"
        current_app.logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@bp.route('/analyze-structured', methods=['POST'])
def analyze_structured():
    """Endpoint for analyzing structured JSON data.
    
    Example POST body:
    {
        "data": { 
            "accidentTypeDistribution": [...],
            "vehicleTypeDistribution": [...],
            ...
        }
    }
    """
    request_data = request.json
    
    if not request_data or 'data' not in request_data:
        return jsonify({'error': 'Data is required'}), 400
    
    data = request_data['data']
    
    try:
        # First test API connection
        connection_ok, connection_msg = AIService.check_api_connection()
        if not connection_ok:
            current_app.logger.error(f"API connection test failed: {connection_msg}")
            return jsonify({'error': f"OpenAI API connection error: {connection_msg}"}), 503
            
        # Generate analysis using the existing JSON insights method
        analysis = AIService.generate_insights_from_json(data)
        
        # Check if the analysis is an error message
        if analysis.startswith("Error generating insights:"):
            return jsonify({'error': analysis}), 500
            
        return jsonify({
            'analysis': analysis,
            'method': 'standard'
        })
    except Exception as e:
        error_msg = f"Error in analyze_structured: {str(e)}"
        current_app.logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@bp.route('/system-check', methods=['GET'])
def system_check():
    """Check system connections and configuration.
    
    This endpoint checks:
    1. Database connection
    2. OpenAI API connection
    3. Environment variables
    4. Model configuration
    
    Returns diagnostic information about the system status.
    """
    results = {
        "database": {"status": "unknown", "message": ""},
        "openai_api": {"status": "unknown", "message": ""},
        "environment": {"status": "unknown", "variables": {}},
        "models": {"status": "unknown", "configuration": {}}
    }
    
    # Check database connection
    try:
        # Get database connection parameters
        db_protocol = current_app.config.get('DB_PROTOCOL', 'postgresql://')
        db_user = current_app.config.get('DB_USER', 'postgres')
        db_password = current_app.config.get('DB_PASSWORD', 'postgres')
        db_host = current_app.config.get('DB_HOST', 'localhost')
        db_port = current_app.config.get('DB_PORT', '5432')
        db_name = current_app.config.get('DB_NAME', 'insights')
        
        conn = psycopg2.connect(
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port,
                dbname=db_name,
                connect_timeout=10
            )
        # Mask password for logs
        masked_url = f"{db_protocol}{db_user}:****@{db_host}:{db_port}/{db_name}"
        current_app.logger.info(f"Testing database connection to: {masked_url}")
        
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        
        results["database"]["status"] = "ok"
        results["database"]["message"] = "Successfully connected to database"
        results["database"]["connection"] = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user
        }
    except Exception as e:
        results["database"]["status"] = "error"
        results["database"]["message"] = f"Database connection error: {str(e)}"
        current_app.logger.error(f"Database check failed: {str(e)}")
    
    # Check OpenAI API connection and models
    try:
        connection_ok, connection_msg = AIService.check_api_connection()
        results["openai_api"]["status"] = "ok" if connection_ok else "error"
        results["openai_api"]["message"] = connection_msg
        
        if connection_ok:
            # Get configured models
            client = AIService.get_client()
            available_models = [m.id for m in client.models.list().data]
            
            # Check each model tier
            model_config = {
                "economy": {
                    "configured": current_app.config.get('ECONOMY_MODEL'),
                    "available": current_app.config.get('ECONOMY_MODEL') in available_models
                },
                "standard": {
                    "configured": current_app.config.get('STANDARD_MODEL'),
                    "available": current_app.config.get('STANDARD_MODEL') in available_models
                },
                "premium": {
                    "configured": current_app.config.get('PREMIUM_MODEL'),
                    "available": current_app.config.get('PREMIUM_MODEL') in available_models
                }
            }
            
            # Check current model configuration
            current_model = current_app.config.get('SQL_MODEL')
            use_schema_aware = current_app.config.get('USE_SCHEMA_AWARE_MODEL', False)
            cost_tier = current_app.config.get('COST_TIER', 'economy')
            
            results["models"]["configuration"] = {
                "current_model": current_model,
                "use_schema_aware": use_schema_aware,
                "cost_tier": cost_tier,
                "model_tiers": model_config
            }
            
            # Set overall model status
            all_models_available = all(tier["available"] for tier in model_config.values())
            current_model_available = current_model in available_models
            
            if all_models_available and current_model_available:
                results["models"]["status"] = "ok"
                results["models"]["message"] = "All configured models are available"
            elif not current_model_available:
                results["models"]["status"] = "error"
                results["models"]["message"] = f"Current model {current_model} is not available"
            else:
                results["models"]["status"] = "warning"
                results["models"]["message"] = "Some configured models are not available"
                
    except Exception as e:
        results["openai_api"]["status"] = "error"
        results["openai_api"]["message"] = f"API check error: {str(e)}"
        current_app.logger.error(f"OpenAI API check failed: {str(e)}")
    
    # Check environment variables
    env_vars = [
        "DB_PROTOCOL", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME",
        "OPENAI_API_KEY", "SCHEMA_FOLDER",
        "ECONOMY_MODEL", "STANDARD_MODEL", "PREMIUM_MODEL",
        "USE_SCHEMA_AWARE_MODEL", "SQL_MODEL", "COST_TIER"
    ]
    all_present = True
    
    for var in env_vars:
        # Check both environment and app config
        env_value = os.environ.get(var)
        config_value = current_app.config.get(var)
        
        # Mask sensitive values
        display_value = "[SET]" if env_value else "[NOT SET]"
        if var == "DB_PASSWORD" and env_value:
            display_value = "[PASSWORD SET]"
        elif var == "OPENAI_API_KEY" and env_value:
            display_value = "[API KEY SET]"
            
        results["environment"]["variables"][var] = {
            "in_env": env_value is not None,
            "in_config": config_value is not None,
            "value": display_value
        }
        
        if not config_value:
            all_present = False
    
    results["environment"]["status"] = "ok" if all_present else "warning"
    if not all_present:
        results["environment"]["message"] = "Some environment variables are missing"
    else:
        results["environment"]["message"] = "All required environment variables are set"
    
    return jsonify(results) 