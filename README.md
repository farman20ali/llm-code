# AI-Powered SQL & Data Insights System

A Flask application that provides AI-powered insights using natural language processing, SQL analysis, and data visualization.

## Features

- Natural language to SQL conversion with insights
- Direct SQL analysis with AI explanations
- JSON data analysis for visualizations and insights
- OpenAI integration for advanced natural language processing
- PostgreSQL database integration

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the root directory with the following variables:
```
SECRET_KEY=your_secret_key_here
DATABASE_URL=postgresql://postgres:postgres@localhost/insights
OPENAI_API_KEY=your_openai_api_key_here
SCHEMA_FOLDER=scripts
```

5. Set up the PostgreSQL database:
```
createdb insights  # Create the database
```

6. Run the database initialization script to execute SQL files:
```
python scripts/init_db.py
```

## Running the Application

Start the Flask development server:
```
python app.py
```

This will display usage information and start the server at `http://localhost:5000`.

## Testing the API

A test script is provided to test the API endpoints:

```
# Test all endpoints
python test_api.py

# Test a specific endpoint
python test_api.py --endpoint ask --question "How many accidents occurred last week?"
python test_api.py --endpoint sql --sql "SELECT COUNT(*) FROM accidents"
python test_api.py --endpoint json --data-file data.json
```

## API Endpoints

### 1. Natural Language to SQL with Insights

**Endpoint:** `POST /api/ask`

**Request Body:**
```json
{
    "question": "How many car accidents were reported in the last week?"
}
```

**Response:**
```json
{
    "sql": "SELECT COUNT(*) as accident_count FROM accidents WHERE accident_date >= NOW() - INTERVAL '7 days' AND vehicle_type = 'Car'",
    "columns": ["accident_count"],
    "rows": [[12]],
    "insight": "There were 12 car accidents reported in the last week."
}
```

### 2. SQL to Insights

**Endpoint:** `POST /api/sql-insights`

**Request Body:**
```json
{
    "sql": "SELECT vehicle_type, COUNT(*) as count FROM accidents GROUP BY vehicle_type ORDER BY count DESC"
}
```

**Response:**
```json
{
    "sql": "SELECT vehicle_type, COUNT(*) as count FROM accidents GROUP BY vehicle_type ORDER BY count DESC",
    "columns": ["vehicle_type", "count"],
    "rows": [["Car", 45], ["Motorcycle", 12], ["Bicycle", 8]],
    "insight": "Cars are the most common vehicle type involved in accidents with 45 incidents, followed by motorcycles (12) and bicycles (8)."
}
```

### 3. JSON to Insights

**Endpoint:** `POST /api/json-insights`

**Request Body:**
```json
{
    "data": {
        "accidentTypeDistribution": [
            {"label": "Minor Collision", "count": 91, "avgSeverity": 2.01},
            {"label": "Major Collision", "count": 72, "avgSeverity": 2.5},
            {"label": "Vehicle Rollover", "count": 56, "avgSeverity": 2.48}
        ],
        "vehicleTypeDistribution": [
            {"label": "Pedestrian", "count": 79, "avgSeverity": 2.06},
            {"label": "Bicycle", "count": 74, "avgSeverity": 2.27},
            {"label": "Motorbike", "count": 64, "avgSeverity": 2.64}
        ]
    }
}
```

**Response:**
```json
{
    "data": {...},
    "insight": "Analysis of the accident data shows that Minor Collisions are the most common accident type (91 incidents), while Pedestrians are the most frequently involved vehicle type (79 incidents). However, Motorbike accidents have the highest average severity at 2.64."
}
```

## Project Structure

```
├── app/                  # Application package
│   ├── __init__.py       # Application factory
│   ├── routes/           # API routes and views
│   │   ├── main.py       # Main routes
│   │   └── api.py        # API endpoints
│   ├── services/         # Business logic
│   │   ├── ai_service.py # OpenAI integration
│   │   └── sql_service.py# SQL handling
│   ├── static/           # Static files (CSS, JS)
│   └── templates/        # HTML templates
├── scripts/              # SQL scripts and utilities
│   ├── init_db.py        # Database initialization script
│   └── *.sql             # SQL files with database schema
├── .env                  # Environment variables
├── app.py                # Application entry point
├── test_api.py           # API testing script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## License

MIT

##installation:
# Remove old images
docker-compose down
docker system prune -f

# Rebuild with new dependencies
docker-compose build --no-cache
docker-compose up -d
docker logs $(docker ps -q --filter name=web) | cat

# deploying using gcorn
## directory structure
myapp/
├── app.py
├── requirements.txt
├── wsgi.py

# wsgi.py
from app import create_app

app = create_app()

# install dependencies in virtual environment

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# if gunicorn is not installed
pip install gunicorn

# run and test
gunicorn --bind 0.0.0.0:8000 wsgi:app

# deploying as service
create file 
sudo nano /etc/systemd/system/aisql.service

or copy from current directory

to find user type whoami

## content of aisql
----------------------
[Unit]
Description=Gunicorn instance to serve AI SQL Flask App
After=network.target

[Service]
User=root
Group=www-data
WorkingDirectory=/home/farman/farman_ws/llm-code
Environment="PATH=/home/farman/farman_ws/llm-code/venv/bin"
ExecStart=/home/farman/farman_ws/llm-code/venv/bin/gunicorn --workers 3 --bind 0.0.0.0:5000 wsgi:app

[Install]
WantedBy=multi-user.target

---------------------


now run:
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl start aisql
sudo systemctl enable aisql


# deploying using script
chmod +x deploy_flask.sh
./deploy_flask.sh


