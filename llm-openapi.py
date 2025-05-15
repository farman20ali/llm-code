from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from functools import wraps

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://username:password@localhost:5432/irs')
engine = create_engine(DATABASE_URL)

# OpenAI configuration
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY', "your-api-key")
)

def validate_select_only(query):
    """Validate that the query is SELECT only"""
    cleaned_query = query.strip().lower()
    if not cleaned_query.startswith('select'):
        raise ValueError("Only SELECT queries are allowed")
    if any(keyword in cleaned_query for keyword in ['insert', 'update', 'delete', 'drop', 'alter', 'create']):
        raise ValueError("Only SELECT queries are allowed")
    return True

def get_llm_insights(data, query_context):
    """Get insights from LLM based on the query results"""
    system_prompt = """You are an analytics expert. Analyze the following data from an emergency response 
    and accident reporting system. Provide clear, actionable insights. Context: {context}"""
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt.format(context=query_context)},
                {"role": "user", "content": f"Analyze this data and provide insights: {data}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error getting insights: {str(e)}"

@app.route('/api/insights', methods=['POST'])
def get_insights():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
            
        query = data['query']
        context = data.get('context', 'General database query')
        
        # Validate query is SELECT only
        validate_select_only(query)
        
        # Execute query
        with engine.connect() as connection:
            result = connection.execute(text(query))
            query_results = [dict(row) for row in result]
            
        # Get insights from LLM
        insights = get_llm_insights(query_results, context)
        
        return jsonify({
            "data": query_results,
            "insights": insights
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/tables', methods=['GET'])
def get_available_tables():
    """Get list of available tables and their descriptions"""
    query = """
    SELECT 
        table_name,
        obj_description((quote_ident(table_schema) || '.' || quote_ident(table_name))::regclass, 'pg_class') as description
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            tables = [dict(row) for row in result]
        return jsonify({"tables": tables})
    except Exception as e:
        return jsonify({"error": f"Error fetching tables: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

