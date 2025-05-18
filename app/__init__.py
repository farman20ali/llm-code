import os
from flask import Flask
from dotenv import load_dotenv
import json
import logging
from app.routes.sql_routes import sql_bp
from app.routes.query_routes import query_bp
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Schema model config file
SCHEMA_MODEL_CONFIG = 'schema_model_config.json'

def load_model_config():
    """Load model configuration from file or environment variables."""
    # Look for config in root directory
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), SCHEMA_MODEL_CONFIG)
    model_config = {}
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                model_config = json.load(f)
                logger.info(f"Loaded model configuration from {config_file}")
                logger.info(f"Using model: {model_config.get('model_id')}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
    else:
        raise ValueError(f"Config file {config_file} not found")
    
    # Use exact variable names from config file, with environment variable overrides
    return {
        'USE_SCHEMA_AWARE_MODEL': os.environ.get('USE_SCHEMA_AWARE_MODEL', 
            str(model_config.get('use_schema_aware_model', 'true'))).lower() == 'true',
        'SQL_MODEL': os.environ.get('SQL_MODEL', 
            model_config.get('model_id', 'gpt-3.5-turbo')),
        'COST_TIER': os.environ.get('COST_TIER', 
            model_config.get('cost_tier', 'economy')).lower(),
        'TEMPERATURE': float(os.environ.get('TEMPERATURE', 
            str(model_config.get('temperature', '0.0')))),
        'MAX_TOKENS': int(os.environ.get('MAX_TOKENS', 
            str(model_config.get('max_tokens', '2048'))))
    }

def create_app(test_config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    
    # Get individual database connection parameters
    db_protocol = os.environ.get('DB_PROTOCOL', 'postgresql://')
    db_user = os.environ.get('DB_USER', 'postgres')
    db_password = os.environ.get('DB_PASSWORD', 'postgres')
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_port = os.environ.get('DB_PORT', '5432')
    db_name = os.environ.get('DB_NAME', 'insights')
    # Build the database URL from individual components
    db_url = f"{db_protocol}{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    database_uri=os.environ.get('DATABASE_URL',db_url)
    
    # Load model configuration from file
    model_config = load_model_config()
    
    # Default configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        DATABASE_URL=db_url,
        DATABASE_URI=database_uri,
        DB_PROTOCOL=db_protocol,
        DB_USER=db_user,
        DB_PASSWORD=db_password,
        DB_HOST=db_host,
        DB_PORT=db_port,
        DB_NAME=db_name,
        OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY'),
        SCHEMA_FOLDER=os.environ.get('SCHEMA_FOLDER', os.path.join(app.root_path, '..', 'scripts')),
        # SQL model configuration - environment variables override file config
        USE_SCHEMA_AWARE_MODEL=model_config['USE_SCHEMA_AWARE_MODEL'],
        SQL_MODEL=model_config['SQL_MODEL'],
        # Model options by cost tier (from lowest to highest cost)
        ECONOMY_MODEL=os.environ.get('ECONOMY_MODEL', 'gpt-3.5-turbo'),     # Lowest cost, good for basic queries
        STANDARD_MODEL=os.environ.get('STANDARD_MODEL', 'gpt-4o-mini'),      # Mid-tier cost, good balance
        PREMIUM_MODEL=os.environ.get('PREMIUM_MODEL', 'gpt-4o'),            # Higher cost, best quality
        # Default cost tier to use
        COST_TIER=model_config['COST_TIER'],
        TEMPERATURE=model_config['TEMPERATURE'],
        MAX_TOKENS=model_config['MAX_TOKENS'],
        PORT=os.environ.get('PORT', 5000),
    )
    
    # Log database connection info (without password) for debugging
    if app.debug:
        masked_url = f"{db_protocol}{db_user}:****@{db_host}:{db_port}/{db_name}"
        print(f"Database URL: {masked_url}")
    
    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)
    
    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    # Register blueprints
    from app.routes import main, api
    
    app.register_blueprint(main.bp)
    app.register_blueprint(api.bp)
    app.register_blueprint(sql_bp, url_prefix='/api/sql')
    app.register_blueprint(query_bp,url_prefix='/query')
    
    return app 