from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI
from flask import current_app
from sqlalchemy import types
from sqlalchemy.dialects.postgresql import base as postgresql_base


available_tables = [
            "accident_reports",  # Main table
            "accident_types",    # accident_reports.accident_type_id -> accident_types.id
            "patient_victim",    # accident_reports.patient_victim_id -> patient_victim.id
            "vehicle_involved",  # accident_reports.vehicle_involved_id -> vehicle_involved.id
            "weather_condition", # accident_reports.weather_condition -> weather_condition.id
            "visibility",        # accident_reports.visibility -> visibility.id
            "road_surface_condition", # accident_reports.road_surface_condition -> road_surface_condition.id
            "road_type",        # accident_reports.road_type -> road_type.id
            "road_signage",     # accident_reports.road_markings -> road_signage.id
            "preliminary_fault_assessment", # accident_reports.preliminary_fault -> preliminary_fault_assessment.id
            "gender_types",     # accident_reports.gender -> gender_types.id
            "apparent_cause"    # accident_reports.cause -> apparent_cause.id
        ]

class GeometryType(types.UserDefinedType):
    """Custom type for PostGIS geometry columns"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def get_col_spec(self, **kw):
        return "GEOMETRY"

# Register the geometry type with PostgreSQL dialect
postgresql_base.ischema_names['geometry'] = GeometryType

class LangChainService:
    _db = None
    _chain = None

    @staticmethod
    def get_model_by_cost_tier(cost_tier='economy'):
        try:
            cost_tier = cost_tier.lower()
            return {
                'premium': current_app.config.get('PREMIUM_MODEL', 'gpt-4'),
                'standard': current_app.config.get('STANDARD_MODEL', 'gpt-3.5-turbo'),
                'economy': current_app.config.get('ECONOMY_MODEL', 'gpt-3.5-turbo')
            }[cost_tier]
        except KeyError:
            current_app.logger.warning("Invalid cost tier, using economy model")
            return 'gpt-3.5-turbo'

    @classmethod
    def get_db(cls):
        if cls._db is None:
            db_uri = current_app.config.get('DATABASE_URI')
            if not db_uri:
                raise ValueError("DATABASE_URI not configured")

            cls._db = SQLDatabase.from_uri(
                db_uri,
                include_tables=available_tables,
                sample_rows_in_table_info=1,
                engine_args={
                    'connect_args': {'options': '-csearch_path=public'}
                }
            )
        return cls._db

    @classmethod
    def get_chain(cls):
        if cls._chain is None:
            api_key = current_app.config.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not configured")

            llm = ChatOpenAI(
                model_name=cls.get_model_by_cost_tier(),
                temperature=0,
                api_key=api_key
            )

            cls._chain = SQLDatabaseChain.from_llm(
                llm=llm,
                db=cls.get_db(),
                verbose=False,
                return_intermediate_steps=True,
                top_k=3,
                use_query_checker=True
            )
        return cls._chain

    @classmethod
    def generate_sql(cls, question: str) -> dict:
        try:
            chain = cls.get_chain()
            response = chain.invoke({"query": question})

            # Extract SQL from intermediate steps
            sql = None
            if isinstance(response, dict) and 'intermediate_steps' in response:
                steps = response['intermediate_steps']
                if steps and isinstance(steps, list) and len(steps) > 0:
                    first_step = steps[0]
                    if isinstance(first_step, dict):
                        sql = first_step.get('sql_cmd')
                    elif isinstance(first_step, str):
                        sql = first_step

            # Fallback extraction
            if not sql:
                sql = str(response).split('SQLResult:')[-1].strip()

            # Clean SQL output
            if sql:
                sql = sql.strip().replace('\n', ' ').replace('  ', ' ')
                if not sql.endswith(';'):
                    sql += ';'

            return {
                'sql': sql,
                'question': question,
                'status': 'success' if sql else 'partial'
            }

        except Exception as e:
            current_app.logger.error(f"SQL Generation Error: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'question': question,
                'status': 'error',
                'sql': None
            }
    @classmethod
    def get_query_generation_chain(cls):
        """New method specifically for SQL generation without execution"""
        if cls._chain is None:
            api_key = current_app.config.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not configured")

            llm = ChatOpenAI(
                model_name=cls.get_model_by_cost_tier(),
                temperature=0,
                api_key=api_key
            )

            cls._chain = SQLDatabaseChain.from_llm(
                llm=llm,
                db=cls.get_db(),
                verbose=False,
                return_intermediate_steps=True,  # Needed to access raw SQL
                return_direct=False,  # Ensures we get the SQL instead of results
                use_query_checker=True,
                top_k=3
            )
        return cls._chain
    
    @classmethod
    def generate_sql_only(cls, question: str) -> dict:
        """New method that returns only the SQL query without execution"""
        try:
            chain = cls.get_query_generation_chain()
            response = chain.invoke({"query": question})

            # Directly extract SQL from the first intermediate step
            sql = response['intermediate_steps'][0]['sql_cmd']

            # Clean SQL output
            sql = sql.strip().replace('\n', ' ').replace('  ', ' ')
            if not sql.endswith(';'):
                sql += ';'

            return {
                'sql': sql,
                'question': question,
                'status': 'success'
            }

        except Exception as e:
            current_app.logger.error(f"SQL Generation Error: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'question': question,
                'status': 'error',
                'sql': None
            }