import os
import re
import glob
import psycopg2
from flask import current_app

class SQLService:
        # Class variable to cache the schema
    _schema_cache = {}
    """Service for SQL validation and execution."""
    
    import re
from flask import current_app

class SQLService:

    @staticmethod
    def validate_select(sql: str) -> bool:
        """Validate that SQL is a safe single SELECT statement."""
        # 1) Remove markdown fences completely
        sql = re.sub(r"```.*?```", "", sql, flags=re.DOTALL)

        # 2) Trim whitespace
        sql = sql.strip()

        # 3) Remove exactly ONE trailing semicolon (and any following spaces/newlines)
        sql = re.sub(r";+\s*$", "", sql)

        # 4) Reject if any semicolon remains (multiple statements)
        if ";" in sql:
            current_app.logger.warning(f"SQL contains multiple statements: {sql}")
            return False

        # 5) Must start with SELECT as a whole word (case‑insensitive)
        if not re.match(r"(?i)^\s*SELECT\b", sql):
            current_app.logger.warning(f"SQL does not start with SELECT: {sql}")
            return False

        # 6) Forbid dangerous keywords as whole words
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"]
        for word in forbidden:
            if re.search(rf"(?i)\b{word}\b", sql):
                current_app.logger.warning(f"SQL contains forbidden keyword '{word}': {sql}")
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

        # Log the folder path being searched
        current_app.logger.info(f"Searching for schema files in: {folder_path}")
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            current_app.logger.error(f"Schema folder not found: {folder_path}")
            raise ValueError(f"Schema folder not found: {folder_path}")

        # Get all SQL files
        sql_files = glob.glob(os.path.join(folder_path, "*.sql"))
        if not sql_files:
            current_app.logger.error(f"No SQL files found in {folder_path}")
            raise ValueError(f"No SQL files found in {folder_path}")
            
        current_app.logger.info(f"Found {len(sql_files)} SQL files: {[os.path.basename(f) for f in sql_files]}")

        for filepath in sql_files:
            current_app.logger.info(f"Processing schema file: {os.path.basename(filepath)}")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract all CREATE TABLE statements
                creates = re.findall(r"(CREATE TABLE.*?;)", content, re.S | re.I)
                if creates:
                    current_app.logger.info(f"Found {len(creates)} CREATE TABLE statements in {os.path.basename(filepath)}")
                    create_statements.extend(creates)
                
                # Extract all INSERT INTO statements
                inserts = re.findall(r"(INSERT INTO.*?;)", content, re.S | re.I)
                if inserts:
                    current_app.logger.info(f"Found {len(inserts)} INSERT statements in {os.path.basename(filepath)}")
                    insert_statements.extend(inserts)
        
        if not create_statements:
            current_app.logger.error("No CREATE TABLE statements found in schema files")
            raise ValueError("No CREATE TABLE statements found in schema files")
        
        # Combine the extracted statements into one string
        combined = []
        
        # Add a brief domain comment for context
        combined.append("/* Traffic Accident Reporting System Schema and Sample Data */")
        
        # Add CREATE TABLE statements
        combined.append("-- SCHEMA DEFINITIONS (CREATE TABLE statements)")
        combined.extend(create_statements)
        
        # Add INSERT INTO statements
        if insert_statements:
            combined.append("\n-- SAMPLE DATA (INSERT INTO statements)")
            combined.extend(insert_statements)
            
        schema = "\n\n".join(combined)
        current_app.logger.info(f"Generated schema with {len(create_statements)} tables and {len(insert_statements)} sample records")
        
        # Log the first few lines of schema for verification
        preview_lines = schema.split('\n')[:5]
        current_app.logger.info("Schema preview:\n" + "\n".join(preview_lines))
        
        return schema
    
    @staticmethod
    def get_schema(schema_folder: str) -> str:
        """Get database schema, using cache if available.
        
        Args:
            schema_folder: Path to folder containing schema definitions
            
        Returns:
            str: Combined schema and sample data information
        """
        # Check if schema is already in cache
        if schema_folder in SQLService._schema_cache:
            current_app.logger.info(f"Using cached schema from {schema_folder}")
            return SQLService._schema_cache[schema_folder]
        
        # If not in cache, load it and store in cache
        current_app.logger.info(f"Loading schema from {schema_folder}")
        schema = SQLService.load_schema_from_folder(schema_folder)
        
        # Store in cache
        SQLService._schema_cache[schema_folder] = schema
        current_app.logger.info("Schema cached for future use")
        
        return schema
    
        
    @staticmethod
    def _clean_sql_response(raw_sql: str) -> str:
        """Clean up the SQL response from the model.
        
        Args:
            raw_sql: Raw SQL response from the model
            
        Returns:
            str: Cleaned SQL query
        """
        # Strip any markdown code blocks (```sql...```) or backticks
        if raw_sql.startswith("```") and "```" in raw_sql[3:]:
            # Extract content between first and last ```
            first_delim = raw_sql.find("```")
            last_delim = raw_sql.rfind("```")
            if first_delim != last_delim:
                # Extract content between the delimiters
                content_start = raw_sql.find("\n", first_delim) + 1
                content_end = last_delim
                raw_sql = raw_sql[content_start:content_end].strip()
        
        # Remove inline backticks if present
        clean_sql = raw_sql.replace("`", "").strip()
        
        # Ensure it starts with SELECT
        if not clean_sql.upper().startswith("SELECT"):
            raise ValueError("Generated SQL must be a SELECT statement")
        
        return clean_sql
    
    @staticmethod
    def _validate_sql_against_schema(sql: str, schema_folder: str) -> bool:
        """Validate SQL query against the schema.
        
        Args:
            sql: SQL query to validate
            schema_folder: Path to schema folder
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Load schema
            schema = SQLService.get_schema(schema_folder)
            
            # Extract table names from SQL, handling schema-qualified names
            table_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)|JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)'
            tables = re.findall(table_pattern, sql, re.IGNORECASE)
            # Extract just the table names, removing schema prefixes
            tables = [t[1] or t[3] for t in tables if t[1] or t[3]]
            
            # Log found tables
            current_app.logger.info(f"Found tables in query: {tables}")
            
            # Ensure accident_reports is included (case-insensitive)
            if not any('accident_reports' in t.lower() for t in tables):
                current_app.logger.warning("SQL query must include accident_reports table")
                return False
            
            # Extract column names from SQL
            column_pattern = r'\bSELECT\s+(.*?)\s+FROM'
            columns_match = re.search(column_pattern, sql, re.IGNORECASE)
            if columns_match:
                columns = [c.strip() for c in columns_match.group(1).split(',')]
                current_app.logger.info(f"Found columns in query: {columns}")
                
                # Validate columns exist in schema
                for col in columns:
                    # Skip validation for aggregate functions and their aliases
                    if any(func in col.upper() for func in ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(']):
                        current_app.logger.info(f"Skipping validation for aggregate function: {col}")
                        continue
                    # Skip validation for expressions with AS
                    if ' AS ' in col.upper():
                        current_app.logger.info(f"Skipping validation for aliased expression: {col}")
                        continue
                    # Skip validation for * (SELECT *)
                    if col.strip() == '*':
                        current_app.logger.info("Skipping validation for SELECT *")
                        continue
                    # Skip validation for column references with table aliases
                    if '.' in col:
                        # Extract just the column name after the dot
                        col_name = col.split('.')[-1].strip()
                        current_app.logger.info(f"Validating table-qualified column: {col} -> {col_name}")
                        if not any(col_name in line for line in schema.split('\n')):
                            current_app.logger.warning(f"Column {col_name} not found in schema")
                            return False
                        continue
                        
                    # For remaining columns, validate against schema
                    if not any(col.strip() in line for line in schema.split('\n')):
                        current_app.logger.warning(f"Column {col} not found in schema")
                        return False
            
            current_app.logger.info("SQL validation successful")
            return True
            
        except Exception as e:
            current_app.logger.error(f"Error validating SQL against schema: {str(e)}")
            return False

    @staticmethod
    def get_connection():
        """Get a database connection using configuration from current_app.
        
        Returns:
            psycopg2.connection: Database connection object
        """
        try:
            # Get database connection parameters
            db_protocol = current_app.config.get('DB_PROTOCOL', 'postgresql://')
            db_user = current_app.config.get('DB_USER', 'postgres')
            db_password = current_app.config.get('DB_PASSWORD', 'postgres')
            db_host = current_app.config.get('DB_HOST', 'localhost')
            db_port = current_app.config.get('DB_PORT', '5432')
            db_name = current_app.config.get('DB_NAME', 'insights')
            
            # Create connection
            conn = psycopg2.connect(
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port,
                dbname=db_name,
                connect_timeout=10
            )
            
            # Set autocommit to True for read-only operations
            conn.autocommit = True
            
            return conn
            
        except Exception as e:
            current_app.logger.error(f"Database connection error: {str(e)}")
            raise