import os
import re
import glob
import psycopg2
from flask import current_app

class SQLService:
    """Service for SQL validation and execution."""
    
    @staticmethod
    def validate_select(sql: str) -> bool:
        """Validate that SQL is a safe SELECT statement only.
        
        Args:
            sql: The SQL query to validate
            
        Returns:
            bool: True if SQL is valid, False otherwise
        """
        # Check for markdown formatting or code blocks
        if '```' in sql:
            current_app.logger.warning(f"SQL contains markdown code blocks: {sql}")
            # Try to extract SQL from markdown
            lines = sql.split('\n')
            filtered_lines = [line for line in lines if not line.strip().startswith('```')]
            sql = '\n'.join(filtered_lines)
            
        sql = sql.strip().rstrip(';')
        
        # Check for multiple statements
        if ";" in sql:
            current_app.logger.warning(f"SQL contains multiple statements: {sql}")
            return False
        
        upper = sql.upper()
        # Check if it's a SELECT statement
        if not upper.startswith("SELECT"):
            current_app.logger.warning(f"SQL does not start with SELECT: {sql}")
            return False
            
        # Check for forbidden keywords
        for forbidden in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"]:
            if forbidden in upper:
                current_app.logger.warning(f"SQL contains forbidden keyword '{forbidden}': {sql}")
                return False
                
        return True
    
    @staticmethod
    def run_sql(query: str):
        """Execute SQL query against PostgreSQL.
        
        Args:
            query: The SQL query to execute
            
        Returns:
            tuple: (column_names, rows)
        """
        conn = None
        try:
            # Get individual database connection parameters from config
            db_protocol = current_app.config.get('DB_PROTOCOL', 'postgresql://')
            db_user = current_app.config.get('DB_USER', 'postgres')
            db_password = current_app.config.get('DB_PASSWORD', 'postgres')
            db_host = current_app.config.get('DB_HOST', 'localhost')
            db_port = current_app.config.get('DB_PORT', '5432')
            db_name = current_app.config.get('DB_NAME', 'irs')
            
            # Build connection string from individual components
            # db_url = f"{db_protocol}{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            
            # Log connection attempt with masked password
            masked_url = f"postgresql://{db_user}:****@{db_host}:{db_port}/{db_name}"
            current_app.logger.info(f"Connecting to database: {masked_url}")

            # ✅ Best practice: use keyword arguments
            conn = psycopg2.connect(
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port,
                dbname=db_name,
                connect_timeout=10
            )

            with conn.cursor() as cur:
                cur.execute(query)
                cols = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                return cols, rows
        except psycopg2.Error as e:
            error_msg = f"Database error executing query: {query}\nError: {str(e)}"
            current_app.logger.error(error_msg)
            
            # Provide more helpful error message
            raise RuntimeError(f"Database error: {str(e)}")
        finally:
            if conn is not None:
                conn.close()
    

    @staticmethod
    def load_schema_from_folder(folder_path: str) -> str:
        """Load only CREATE TABLE and INSERT INTO statements from SQL files in a folder.
        
        Args:
            folder_path: Path to the folder containing SQL files
            
        Returns:
            str: Combined schema and sample data (CREATE + INSERT statements)
        """
        create_statements = []
        insert_statements = []

        for filepath in glob.glob(os.path.join(folder_path, "*.sql")):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract all CREATE TABLE statements
                creates = re.findall(r"(CREATE TABLE.*?;)", content, re.S | re.I)
                create_statements.extend(creates)
                
                # Extract all INSERT INTO statements
                inserts = re.findall(r"(INSERT INTO.*?;)", content, re.S | re.I)
                insert_statements.extend(inserts)
        
        # Combine the extracted statements into one string
        combined = []
        
        # Add a brief domain comment for context (optional)
        combined.append("/* Traffic Accident Reporting System Schema and Sample Data */")
        
        # Add CREATE TABLE statements
        combined.append("-- SCHEMA DEFINITIONS (CREATE TABLE statements)")
        combined.extend(create_statements)
        
        # Add INSERT INTO statements
        if insert_statements:
            combined.append("\n-- SAMPLE DATA (INSERT INTO statements)")
            combined.extend(insert_statements)
        print("generated schema ")
        return "\n\n".join(combined)