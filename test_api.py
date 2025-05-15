#!/usr/bin/env python3
import argparse
import json
import requests
import sys

def test_ask(base_url, question):
    """Test the natural language to SQL endpoint."""
    endpoint = f"{base_url}/api/ask"
    payload = {"question": question}
    
    print(f"\n===== Testing /api/ask =====")
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        print(f"\nResponse (Status: {response.status_code}):")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"\nError: {str(e)}")

def test_sql_insights(base_url, sql):
    """Test the SQL to insights endpoint."""
    endpoint = f"{base_url}/api/sql-insights"
    payload = {"sql": sql}
    
    print(f"\n===== Testing /api/sql-insights =====")
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        print(f"\nResponse (Status: {response.status_code}):")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"\nError: {str(e)}")

def test_json_insights(base_url, data_file=None):
    """Test the JSON to insights endpoint."""
    endpoint = f"{base_url}/api/json-insights"
    
    # Sample data if no file provided
    if data_file:
        with open(data_file, 'r') as f:
            data = json.load(f)
    else:
        data = {
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
    
    payload = {"data": data}
    
    print(f"\n===== Testing /api/json-insights =====")
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        print(f"\nResponse (Status: {response.status_code}):")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"\nError: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Test the AI SQL and Data Insights API')
    parser.add_argument('--url', default='http://localhost:5000', help='Base URL of the API')
    parser.add_argument('--endpoint', choices=['ask', 'sql', 'json', 'all'], default='all', help='Endpoint to test')
    parser.add_argument('--question', default='How many accidents occurred last month?', help='Question for the /ask endpoint')
    parser.add_argument('--sql', default='SELECT COUNT(*) FROM accidents', help='SQL for the /sql-insights endpoint')
    parser.add_argument('--data-file', help='JSON file for the /json-insights endpoint')
    
    args = parser.parse_args()
    
    if args.endpoint == 'ask' or args.endpoint == 'all':
        test_ask(args.url, args.question)
    
    if args.endpoint == 'sql' or args.endpoint == 'all':
        test_sql_insights(args.url, args.sql)
    
    if args.endpoint == 'json' or args.endpoint == 'all':
        test_json_insights(args.url, args.data_file)

if __name__ == '__main__':
    main() 