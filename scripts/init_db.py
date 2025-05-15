import os
import sys
import psycopg2
import glob

def init_db():
    """Initialize database using SQL scripts."""
    # Get database URL from environment variable
    db_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost/insights')
    
    # Connect to the database
    conn = psycopg2.connect(db_url)
    conn.autocommit = True  # Set autocommit mode
    cursor = conn.cursor()
    
    print("Initializing database...")
    
    # Find and execute all SQL files in the scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sql_files = glob.glob(os.path.join(script_dir, "*.sql"))
    
    for sql_file in sorted(sql_files):
        file_name = os.path.basename(sql_file)
        print(f"Executing {file_name}...")
        
        with open(sql_file, 'r') as f:
            sql_script = f.read()
            try:
                cursor.execute(sql_script)
                print(f"- {file_name} executed successfully")
            except Exception as e:
                print(f"- Error executing {file_name}: {str(e)}")
    
    # Close the connection
    cursor.close()
    conn.close()
    
    print("Database initialization completed.")

if __name__ == '__main__':
    # Load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    init_db() 