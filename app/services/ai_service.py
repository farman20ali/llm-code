import os
from openai import OpenAI
from flask import current_app
import sys
import os.path
import json
import random
import time
import re
from dateutil.parser import parse as parse_date
# Add the parent directory of app to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.services.sql_service import SQLService

# Define the function/tool specs
tools = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "get_all_table_names",
    #         "description": "Returns a list of all available table names in the accident reporting system.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {},
    #             "required": []
    #         }
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": "Returns the schema of a given table, including columns and foreign key references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to retrieve schema for."
                    }
                },
                "required": ["table_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_multiple_table_schema",
            "description": "Returns the schema for multiple tables at once, including their columns, foreign keys, and relationships.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_names": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of table names to retrieve schemas for."
                    }
                },
                "required": ["table_names"]
            }
        }
    }
]
class AIService:
    """Service for interacting with OpenAI API."""
    
    # Class variable to cache the client
    _client = None
    _compressed_schema = None
    _compressed_schema_hashmap = None

    @staticmethod
    def get_client():
        """Get OpenAI client with API key from environment."""
        if AIService._client is not None:
            return AIService._client
            
        try:
            api_key = current_app.config.get('OPENAI_API_KEY')
            if not api_key:
                current_app.logger.error("OpenAI API key is missing from configuration")
                raise ValueError("OpenAI API key not configured")
            
            current_app.logger.info(f"API Key prefix: {api_key[:8]}...")
            
            # Initialize client with minimal configuration
            AIService._client = OpenAI(
                api_key=api_key
            )
            
            # Test the client by listing models
            models = AIService._client.models.list()
            current_app.logger.info(f"Successfully initialized OpenAI client. Found {len(models.data)} models.")
            
            return AIService._client
        except Exception as e:
            current_app.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def check_api_connection():
        """Test the connection to OpenAI API.
        
        Returns:
            tuple: (bool, str) - Success status and message
        """
        try:
            client = AIService.get_client()
            # Make a minimal API call to test the connection
            response = client.models.list()
            return True, "Connection successful"
        except Exception as e:
            error_msg = f"Failed to connect to OpenAI API: {str(e)}"
            current_app.logger.error(error_msg)
            return False, error_msg
    
    
    @staticmethod
    def get_model_by_cost_tier(cost_tier='economy'):
        """Get the appropriate model based on cost tier configuration.
        
        Args:
            cost_tier: The cost tier to use (economy, standard, premium)
            
        Returns:
            str: The model name to use
        """
        try:
            cost_tier = cost_tier.lower()
            if cost_tier == 'premium':
                return current_app.config.get('PREMIUM_MODEL', 'gpt-4o')
            elif cost_tier == 'standard':
                return current_app.config.get('STANDARD_MODEL', 'gpt-4o-mini')
            else:  # economy is the default
                return current_app.config.get('ECONOMY_MODEL', 'gpt-3.5-turbo')
        except Exception as e:
            current_app.logger.warning(f"Error determining model from cost tier: {str(e)}. Using economy model.")
            return 'gpt-3.5-turbo'
    
    
    def generate_fine_tuning_data_batch(self, num_examples=10, output_file=None, compress_schema=True):
        """Generate fine-tuning data in JSONL format with batched SQL generation."""
        try:
            current_app.logger.info(f"inside generate_fine_tuning_data_batch")
            schema_folder = current_app.config.get('SCHEMA_FOLDER')

            if not output_file:
                timestamp = int(time.time())
                output_file = os.path.join(schema_folder, f"fine_tuning_data_{timestamp}.jsonl")

            # Load and compress schema
            schema = SQLService.get_schema(schema_folder)
            if compress_schema:
                current_app.logger.info("Compressing schema to reduce token usage")
                schema = self._compress_schema(schema)
            
            # Log schema for verification
            if schema:
                with open(os.path.join(schema_folder, f"schema_generated.txt"), 'w') as f:
                    f.write(schema)
                    
            # More focused sample queries
            sample_queries = [
                "Show me the total number of accidents in Karachi",
                "How many accidents involved motorcycles in Karachi?",
                "Count accidents by severity level in Karachi",
                "Find accidents with more than 3 casualties",
                "Show accidents with hospital transfers in the last week",
                "List all accidents involving pedestrians",
                "Count accidents by weather condition",
                "What areas have the highest number of accidents?",
                "Show accidents that occurred during rainy weather",
                "Which vehicle type is involved in the most accidents?",
                "List all dispatches that took more than 30 minutes",
                "Show accidents on roads with poor visibility"
            ]

            # Add variations with different phrasings
            all_queries = []
            for q in sample_queries:
                all_queries += [
                    q,
                    q.replace("Show", "List"),
                    q.replace("Find", "Get"),
                    f"I need to {q.lower()}",
                    f"Could you {q.lower()}?",
                    f"Please provide {q.lower()}",
                    f"Give me {q.lower()}"
                ]

            if len(all_queries) > num_examples:
                all_queries = random.sample(all_queries, num_examples)
            elif len(all_queries) < num_examples:
                while len(all_queries) < num_examples:
                    base = random.choice(sample_queries)
                    all_queries.append(f"Please tell me about {base.lower()}")
 
            # Batch in groups of 5
            batch_size = 5
            current_app.logger.info(f"Starting batch processing with size: {batch_size}")
            all_examples = []
            
            for i in range(0, len(all_queries), batch_size):
                batch_queries = all_queries[i:i + batch_size]
                current_app.logger.info(f"Processing batch {i//batch_size + 1} of {(len(all_queries) + batch_size - 1)//batch_size}")
                current_app.logger.info(f"Batch queries: {batch_queries}")
                
                examples = self._generate_batch_sql_examples(batch_queries, schema)
                
                # Validate examples before adding
                valid_examples = []
                for example in examples:
                    sql = example["messages"][-1]["content"]  # Get the SQL from the assistant's message
                    try:
                        clean_sql = SQLService._clean_sql_response(sql)
                        if SQLService._validate_sql_against_schema(clean_sql, schema_folder):
                            valid_examples.append(example)
                        else:
                            current_app.logger.warning(f"Example failed schema validation, regenerating...")
                            # Regenerate single example
                            question = example["messages"][1]["content"]  # Get the question
                            retry_examples = self._generate_batch_sql_examples([question], schema)
                            if retry_examples and SQLService._validate_sql_against_schema(
                                SQLService._clean_sql_response(retry_examples[0]["messages"][-1]["content"]), 
                                schema_folder
                            ):
                                valid_examples.append(retry_examples[0])
                    except Exception as e:
                        current_app.logger.error(f"Error validating example: {str(e)}")
                        continue
                
                all_examples.extend(valid_examples)
                current_app.logger.info(f"Added {len(valid_examples)} valid examples from batch")

            # Save to file
            with open(output_file, 'w') as f:
                for ex in all_examples:
                    f.write(json.dumps(ex) + '\n')

            current_app.logger.info(f"Saved {len(all_examples)} examples to {output_file}")
            current_app.logger.info("Fine-tuning data generation completed successfully")
            return output_file

        except Exception as e:
            current_app.logger.error(f"Error generating fine-tuning data: {str(e)}")
            return None

    def _generate_batch_sql_examples(self, queries, schema):
        """Generate SQL queries for a batch of natural language questions."""
        system_prompt = (
            "You are a PostgreSQL SQL assistant for a traffic accident reporting system. "
            "Generate specific and optimized SQL SELECT statements based on user questions. "
            "ALWAYS use EXACT column names from the provided schema. "
            "Always include relevant WHERE clauses and aggregations as needed. "
            "Never use SELECT * unless specifically requested. "
            "Use correct table and column names from the schema. "
            "Do not include explanations or comments."
            f"Here is the Schema : ${schema}"
        )

        # Create a focused batch prompt with schema
        batch_prompt = (
            "Generate specific and optimized SQL queries for the following questions. "
            "Use EXACT column names from the schema. "
            "For each question, provide a detailed query with appropriate JOINs, WHERE clauses, and aggregations:\n\n"
            "Questions:\n"
        )
        for idx, q in enumerate(queries):
            batch_prompt += f"Question {idx+1}: {q}\n"

        messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": batch_prompt}
        ]
        current_app.logger.info(f" messages : ${messages}")
        client = self.get_client()
        # Use standard model from config for batch generation
        cost_tier = current_app.config.get('COST_TIER', 'economy')
        model_name = AIService.get_model_by_cost_tier(cost_tier)
        
        current_app.logger.info(f"Generating batch SQL examples using model: {model_name}")
        current_app.logger.info(f"Processing {len(queries)} queries in batch")

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=2048
        )

        # Log token usage
        current_app.logger.info(f"Batch generation token usage - Prompt: {response.usage.prompt_tokens}, "
                             f"Completion: {response.usage.completion_tokens}, "
                             f"Total: {response.usage.total_tokens}")

        response_text = response.choices[0].message.content.strip()
        sql_blocks = response_text.split("SQL ")

        # Extract SQLs
        extracted_sql = []
        for block in sql_blocks:
            if not block.strip():
                continue
            parts = block.split(":", 1)
            if len(parts) < 2:
                continue
            sql = parts[1].strip()
            
            # Validate SQL is not a default query
            if "SELECT * FROM accident_reports LIMIT 10" in sql:
                # Generate a more specific query based on the question
                question = queries[len(extracted_sql)]
                current_app.logger.info(f"Regenerating specific query for: {question}")
                retry_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate a specific SQL query for: {question}"}
                ]
                retry_response = client.chat.completions.create(
                    model=model_name,
                    messages=retry_messages,
                    temperature=0,
                    max_tokens=1024
                )
                sql = retry_response.choices[0].message.content.strip()
            
            # Validate SQL against schema
            try:
                clean_sql = SQLService._clean_sql_response(sql)
                if not SQLService._validate_sql_against_schema(clean_sql, current_app.config.get('SCHEMA_FOLDER')):
                    current_app.logger.warning(f"Generated SQL failed schema validation, regenerating...")
                    question = queries[len(extracted_sql)]
                    retry_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "system", "content": f"Schema:\n{schema}"},
                        {"role": "user", "content": f"Generate a specific SQL query for: {question}"}
                    ]
                    retry_response = client.chat.completions.create(
                        model=model_name,
                        messages=retry_messages,
                        temperature=0,
                        max_tokens=1024
                    )
                    sql = retry_response.choices[0].message.content.strip()
            except Exception as e:
                current_app.logger.error(f"Error validating SQL: {str(e)}")
                continue
                
            extracted_sql.append(sql)
            current_app.logger.info(f"Generated SQL for question {len(extracted_sql)}: {sql}")

        # Pad if missing with specific queries
        while len(extracted_sql) < len(queries):
            question = queries[len(extracted_sql)]
            current_app.logger.info(f"Generating missing query for: {question}")
            retry_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Schema:\n{schema}"},
                {"role": "user", "content": f"Generate a specific SQL query for: {question}"}
            ]
            retry_response = client.chat.completions.create(
                model=model_name,
                messages=retry_messages,
                temperature=0,
                max_tokens=1024
            )
            sql = retry_response.choices[0].message.content.strip()
            extracted_sql.append(sql)

        examples = []
        for user_msg, sql_msg in zip(queries, extracted_sql):
            examples.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": sql_msg}
                ]
            })
            current_app.logger.info(f"Created example for: {user_msg}")

        return examples


    def generate_fine_tuning_data(self, num_examples=15, output_file=None, compress_schema=True):
        """Generate fine-tuning data to create a schema-aware model."""
        try:
            current_app.logger.info(f"inside generate_fine_tuning_data")
            schema_folder = current_app.config.get('SCHEMA_FOLDER')

            if not output_file:
                timestamp = int(time.time())
                output_file = os.path.join(schema_folder, f"fine_tuning_data_{timestamp}.jsonl")

            # Load and compress schema
            schema = SQLService.get_schema(schema_folder)
            if compress_schema:
                current_app.logger.info("Compressing schema to reduce token usage")
                schema = self._compress_schema(schema)
            
            # Log schema for verification
            current_app.logger.info("Schema being used for fine-tuning is saved in file")
            if schema:
                with open(os.path.join(schema_folder, f"schema_generated.txt"), 'w') as f:
                    f.write(schema)

            # More diverse and specific sample queries with exact column names
            sample_queries = [
                "Show me the total number of accidents in Karachi",
                "How many accidents involved motorcycles in Karachi?",
                "Count accidents by severity level in Karachi",
                "Find accidents with more than 3 casualties",
                "Show accidents with hospital transfers in the last week",
                "List all accidents involving pedestrians",
                "Count accidents by weather condition",
                "What areas have the highest number of accidents?",
                "Show accidents that occurred during rainy weather",
                "Which vehicle type is involved in the most accidents?",
                "List all dispatches that took more than 30 minutes",
                "Show accidents on roads with poor visibility"
            ]

            # Add variations with different phrasings
            all_queries = []
            for query in sample_queries:
                all_queries.append(query)
                # Add variations of each query
                all_queries.append(query.replace("Show", "List"))
                all_queries.append(query.replace("Find", "Get"))
                all_queries.append(f"I need to {query.lower()}")
                all_queries.append(f"Could you {query.lower()}?")
                all_queries.append(f"Please provide {query.lower()}")
                all_queries.append(f"Give me {query.lower()}")

            # Select random subset if we have more than needed
            if len(all_queries) > num_examples:
                all_queries = random.sample(all_queries, num_examples)
            elif len(all_queries) < num_examples:
                # If we need more examples, create variations of existing queries
                while len(all_queries) < num_examples:
                    base_query = random.choice(sample_queries)
                    variation = f"Please tell me about {base_query.lower().replace('show', '').replace('find', '').replace('list', '')}"
                    all_queries.append(variation)

            # Generate fine-tuning examples
            examples = []
            
            # Create a single batch request for all queries
            batch_prompt = (
                "Generate specific and optimized SQL queries for the following questions. "
                "Use EXACT column names from the schema. "
                "For each question, provide a detailed query with appropriate JOINs, WHERE clauses, and aggregations:\n\n"
                "Questions:\n"
            )
            for i, query in enumerate(all_queries):
                batch_prompt += f"Question {i+1}: {query}\n"

            # Make a single API call for all examples
            cost_tier = current_app.config.get('COST_TIER', 'economy')
            model_name = AIService.get_model_by_cost_tier(cost_tier)
            current_app.logger.info(f"Using model: {model_name} (cost tier: {cost_tier}")
       
            
            system_prompt_global = (
                "You are a PostgreSQL SQL assistant for a traffic accident reporting system. "
                "The main table is 'accident_reports' which contains all accident data. "
                "containing Columns: report_id(serial4), latitude(numeric(9, 6) NULL), longitude(numeric(9, 6) NULL), accident_location(varchar(255) NULL), gis_coordinates(GEOMETRY(Point, 4326) NULL), user_id(int4), num_affecties(int4), age(int4), created_at(timestamp), status(varchar(50) DEFAULT), severity(int4), image_uri(text), audio_uri(text), video_uri(text), description(text), officer_name(VARCHAR(255)), officer_designation(VARCHAR(100)), officer_contact_no(VARCHAR(20)), officer_notes(TEXT), weather_condition(int4), visibility(int4), road_surface_condition(int4), road_type(int4), road_markings(int4), preliminary_fault(int4), gender(int4), cause(int4), vehicle_involved_id(int4), patient_victim_id(int4), accident_type_id(int4)\n\n"
                "Other tables are lookup tables that provide additional information through foreign key relationships: \n"
                "- lov_table: accident_reports column -> lov_table column (descriptive column of lov_table)\n"
                "- accident_types: accident_type_id -> accident_types.id (label)\n"
                "- patient_victim: patient_victim_id -> patient_victim.id (label)\n"
                "- vehicle_involved: vehicle_involved_id -> vehicle_involved.id (label)\n"
                "- weather_condition: weather_condition -> weather_condition.id (condition)\n"
                "- visibility: visibility -> visibility.id (level)\n"
                "- road_surface_condition: road_surface_condition -> road_surface_condition.id (condition)\n"
                "- road_type: road_type -> road_type.id (type)\n"
                "- road_signage: road_markings -> road_signage.id (status)\n"
                "- preliminary_fault_assessment: preliminary_fault -> preliminary_fault_assessment.id (fault)\n"
                "- gender_types: gender -> gender_types.id (label)\n"
                "- apparent_cause: cause -> apparent_cause.id (cause)\n\n"
                "IMPORTANT: You already have all the schema information for accident_reports and lookup tables above. "
                "DO NOT call get_table_schema or get_multiple_table_schema for schema information. "
                "Only use these tools if you need sample data from lookup tables to understand their values. "
                "Always return optimized SQL SELECT queries without comments or explanation. "
                "Use proper WHERE, JOINs, and avoid SELECT * unless asked explicitly. "
                "When joining tables, use the correct foreign key relationships listed above. "
                )

            messages = [
                {"role": "system", "content": system_prompt_global},
                {"role": "user", "content": batch_prompt}
            ]
            current_app.logger.info(f" messages : {messages}")
            client = self.get_client()
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=2048
            )

            response_text = response.choices[0].message.content.strip()
            sql_blocks = response_text.split("SQL ")

            # Extract SQLs
            extracted_queries = []
            for block in sql_blocks:
                if not block.strip():
                    continue
                parts = block.split(":", 1)
                if len(parts) < 2:
                    continue
                sql = parts[1].strip()
                # Validate SQL is not a default query
                if "SELECT * FROM accident_reports LIMIT 10" in sql:
                    # Generate a more specific query based on the question
                    question = all_queries[len(extracted_queries)]
                    retry_messages = [
                        {"role": "system", "content": system_prompt_global},
                        {"role": "user", "content": f"Generate a specific SQL query for: {question}"}
                    ]
                    retry_response = client.chat.completions.create(
                        model=model_name,
                        messages=retry_messages,
                        temperature=0,
                        max_tokens=1024
                    )
                    sql = retry_response.choices[0].message.content.strip()
                extracted_queries.append(sql)

            # Ensure we have enough queries
            if len(extracted_queries) < len(all_queries):
                current_app.logger.warning(f"Only extracted {len(extracted_queries)} SQL queries for {len(all_queries)} questions")
                # Generate specific queries for missing ones
                while len(extracted_queries) < len(all_queries):
                    question = all_queries[len(extracted_queries)]
                    retry_messages = [
                        {"role": "system", "content": system_prompt_global},
                        {"role": "user", "content": f"Generate a specific SQL query for: {question}"}
                    ]
                    retry_response = client.chat.completions.create(
                        model=model_name,
                        messages=retry_messages,
                        temperature=0,
                        max_tokens=1024
                    )
                    sql = retry_response.choices[0].message.content.strip()
                    extracted_queries.append(sql)

            # Create the training examples
            for i, (query, sql) in enumerate(zip(all_queries, extracted_queries)):
                # Validate SQL against schema before creating example
                if not SQLService._validate_sql_against_schema(sql, schema_folder):
                    current_app.logger.warning(f"Generated SQL for question {i+1} failed schema validation, regenerating...")
                    retry_messages = [
                        {"role": "system", "content": system_prompt_global},
                        {"role": "user", "content": f"Generate a specific SQL query for: {query}"}
                    ]
                    retry_response = client.chat.completions.create(
                        model=model_name,
                        messages=retry_messages,
                        temperature=0,
                        max_tokens=1024
                    )
                    sql = retry_response.choices[0].message.content.strip()
                
                # Create fine-tuning example
                example = {
                    "messages": [
                        {"role": "system", "content": system_prompt_global},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": sql}
                    ]
                }
                
                examples.append(example)
                current_app.logger.info(f"Created example {i+1}/{len(all_queries)}: {query}")

            # Save to file
            with open(output_file, 'w') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')

            current_app.logger.info(f"Saved {len(examples)} fine-tuning examples to {output_file}")
            current_app.logger.info("Token usage significantly reduced by using batch approach")
            return output_file

        except Exception as e:
            current_app.logger.error(f"Error generating fine-tuning data: {str(e)}")
            return None
    
    def create_fine_tuned_model(self, training_file_path, suffix="accident-sql-generator"):
        """Create a fine-tuned model from training data.
        
        Args:
            training_file_path: Path to training data file
            suffix: Suffix to add to the model name
            
        Returns:
            str: The model ID if successful, None otherwise
        """
        try:
            current_app.logger.info(f"inside create_fine_tuned_model")
            
            cost_tier = current_app.config.get('COST_TIER', 'economy')
            base_model = AIService.get_model_by_cost_tier(cost_tier)
            current_app.logger.info(f"Using model: {base_model} (cost tier: {cost_tier})")
       
            client = self.get_client()
            current_app.logger.info(f"Creating fine-tuned model from {training_file_path}")
            
            # Upload training file
            with open(training_file_path, 'rb') as f:
                current_app.logger.info("Uploading training file...")
                training_file = client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            
            # Create fine-tuning job
            current_app.logger.info(f"Creating fine-tuning job with file ID: {training_file.id}")
            # Use economy model from config as base model
            job = client.fine_tuning.jobs.create(
                training_file=training_file.id,
                model=base_model,  # Base model to fine-tune
                suffix=suffix
            )
            
            current_app.logger.info(f"Created fine-tuning job: {job.id}")
            
            # Check job status (optional - this can take a while)
            current_app.logger.info("Checking initial job status...")
            job_status = client.fine_tuning.jobs.retrieve(job.id)
            current_app.logger.info(f"Job status: {job_status.status}")
            
            # Get the model ID from the job
            # The model ID will be in format "ft:gpt-3.5-turbo-0125:[org]:custom:[suffix]-[timestamp]"
            model_id = f"ft:{job_status.model if job_status.model else base_model}"
            
            # Important: Note that the fine-tuning job will continue running in the 
            # background and may take 1-4 hours to complete
            current_app.logger.info(f"Model ID will be: {model_id}")
            current_app.logger.info("Fine-tuning job is running and may take 1-4 hours to complete")
            
            return model_id
            
        except Exception as e:
            current_app.logger.error(f"Error creating fine-tuned model: {str(e)}")
            return None
    
    
    @staticmethod
    def generate_sql_from_question(question: str, schema_folder: str) -> str:
        """Generate SQL query from natural language question."""
        client = AIService.get_client()
        # Get all available tables first
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
        system_prompt_global = (
            "You are a PostgreSQL SQL assistant for a traffic accident reporting system. "
            "The main table is 'accident_reports' which contains all accident data"
            "containing Columns: report_id(serial4), latitude(numeric(9, 6) NULL), longitude(numeric(9, 6) NULL), accident_location(varchar(255) NULL), gis_coordinates(GEOMETRY(Point, 4326) NULL), user_id(int4), num_affecties(int4), age(int4), created_at(timestamp), status(varchar(50) DEFAULT), severity(int4), image_uri(text), audio_uri(text), video_uri(text), description(text), officer_name(VARCHAR(255)), officer_designation(VARCHAR(100)), officer_contact_no(VARCHAR(20)), officer_notes(TEXT), weather_condition(int4), visibility(int4), road_surface_condition(int4), road_type(int4), road_markings(int4), preliminary_fault(int4), gender(int4), cause(int4), vehicle_involved_id(int4), patient_victim_id(int4), accident_type_id(int4)\n\n"
            " The accident_reports table already contains only Karachi data. The accident_location column contains area names within Karachi, not the city name. Do not apply a filter like accident_location ILIKE '%karachi%'.\n"
            "Other tables are lookup tables that provide additional information through foreign key relationships: \n"
            "- lov_table: accident_reports column -> lov_table column (descriptive column of lov_table)\n"
            "- accident_types: accident_type_id -> accident_types.id (label)\n"
            "- patient_victim: patient_victim_id -> patient_victim.id (label)\n"
            "- vehicle_involved: vehicle_involved_id -> vehicle_involved.id (label)\n"
            "- weather_condition: weather_condition -> weather_condition.id (condition)\n"
            "- visibility: visibility -> visibility.id (level)\n"
            "- road_surface_condition: road_surface_condition -> road_surface_condition.id (condition)\n"
            "- road_type: road_type -> road_type.id (type)\n"
            "- road_signage: road_markings -> road_signage.id (status)\n"
            "- preliminary_fault_assessment: preliminary_fault -> preliminary_fault_assessment.id (fault)\n"
            "- gender_types: gender -> gender_types.id (label)\n"
            "- apparent_cause: cause -> apparent_cause.id (cause)\n\n"
            "Use the lookup tables whenever you need descriptive values for aggregations or filters. "
            "Do not fetch schema or sample data from the database using tools. The schema is already provided above. "
            "Focus exclusively on **aggregate/statistical insights** (counts, averages, distributions, trends, top‑N, etc.). "
            "Never return raw accident_reports rows. "
            "Always produce optimized SQL SELECT statements that compute insights, without comments or explanation. "
            "Use proper WHERE, JOINs, GROUP BY, ORDER BY, and avoid SELECT * unless explicitly asked for detail rows."
            "User input may be in Urdu (اردو). Understand the meaning and intent of Urdu queries and respond only with accurate PostgreSQL SQL statements in English, not with explanations or translations."
            "Only return a single SQL query. Do not repeat or include duplicates."
            "Understand Roman Urdu terms such as 'car', 'bus', 'bike', 'ladki', 'aadmi', 'marnay wala', etc., and map them to appropriate labels in lookup tables. For example, 'car' means a vehicle of type 'Car' in the vehicle_involved table."
            )
        
        # Get the cost tier from configuration
        cost_tier = current_app.config.get('COST_TIER', 'economy')
        
        # Check if we're using a pre-trained model that already knows the schema
        use_schema_aware_model = current_app.config.get('USE_SCHEMA_AWARE_MODEL', True)
        current_app.logger.info(f"Using schema-aware model: {use_schema_aware_model}")
        
        # Get the appropriate model based on cost tier
        model_name = current_app.config.get('SQL_MODEL', AIService.get_model_by_cost_tier(cost_tier))
        
        # Set token limits based on model and cost tier
        max_tokens = {
            'gpt-3.5-turbo': 1024,  # Economy
            'gpt-4o-mini': 2048,    # Standard
            'gpt-4o': 4096          # Premium
        }.get(model_name, 1024)  # Default to economy limit
        
        # Validate model availability
        available = client.models.list().data
        if not any(m.id == model_name for m in available):
            raise ValueError(f"Model {model_name!r} not found in your account")
        
        current_app.logger.info(f"Using model: {model_name} (cost tier: {cost_tier}, max tokens: {max_tokens})")
        
        if use_schema_aware_model:
            messages = [
                {"role": "system", "content": system_prompt_global},
                {"role": "user", "content": question}
            ]
        else:
            raise ValueError(f"not using schema-aware model: {model_name}")
        
        # Adjust temperature based on model
        temperature = 0.0
        
        # Generate SQL with retry logic
        max_retries = 1  # Reduced to 1 since we have better validation now
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools  # Add the tools to the API call
                )
                
                # Handle tool calls if present
                if response.choices[0].finish_reason == "tool_calls":
                    tool_calls = response.choices[0].message.tool_calls
                    print("call mades : ", tool_calls)
                    
                    # Add the assistant's message with tool calls to the conversation
                    messages.append(response.choices[0].message)
                    
                   
                    
                    for call in tool_calls:
                        function_name = call.function.name
                        arguments = json.loads(call.function.arguments)

                        # Call your actual tool backend here
                        if function_name == "get_all_table_names":
                            tool_result = {
                                "tables": available_tables
                            }

                        elif function_name == "get_table_schema":
                            table_name = arguments["table_name"]
                            if table_name not in available_tables:
                                tool_result = {
                                    "error": f"Table '{table_name}' does not exist."
                                }
                            else:
                                # Get schema from SQLService
                                schema = SQLService.get_schema(schema_folder)
                                AIService._compress_schema(schema)
                                tool_result = AIService._compressed_schema_hashmap.get(table_name, {})

                        elif function_name == "get_multiple_table_schema":
                            table_names = arguments["table_names"]
                            # Check if all requested tables exist
                            invalid_tables = [t for t in table_names if t not in available_tables]
                            if invalid_tables:
                                tool_result = {
                                    "error": f"Tables {', '.join(invalid_tables)} do not exist. "
                                }
                            else:
                                # Get schema from SQLService
                                schema = SQLService.get_schema(schema_folder)
                                AIService._compress_schema(schema)
                                # Get schemas for all requested tables
                                tool_result = {
                                    table_name: AIService._compressed_schema_hashmap.get(table_name, {})
                                    for table_name in table_names
                                }

                        else:
                            tool_result = {
                                "error": f"Unknown function: {function_name}"
                            }

                        # Add tool response in the correct format
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": function_name,
                            "content": json.dumps(tool_result)
                        })

                    # Make another API call with the updated conversation
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                
                # Log token usage
                current_app.logger.info(f"Token usage - Prompt: {response.usage.prompt_tokens}, "
                                     f"Completion: {response.usage.completion_tokens}, "
                                     f"Total: {response.usage.total_tokens}")
                
                raw_sql = response.choices[0].message.content.strip()
                current_app.logger.info(f"Raw SQL generated: {raw_sql}")
                
                try:
                    clean_sql = SQLService._clean_sql_response(raw_sql)
                    current_app.logger.info(f"Cleaned SQL: {clean_sql}")
                except ValueError as e:
                    current_app.logger.warning(f"SQL cleaning failed: {str(e)}")
                    if attempt < max_retries - 1:
                        messages.append({"role": "assistant", "content": raw_sql})
                        messages.append({"role": "user", "content": "The generated SQL is not valid. Please generate a proper SELECT statement."})
                        continue
                    raise
                
                try:
                    if SQLService._validate_sql_against_schema(clean_sql, schema_folder):
                        current_app.logger.info("SQL validation successful")
                        return clean_sql
                    else:
                        current_app.logger.warning("SQL validation failed")
                        if attempt < max_retries - 1:
                            messages.append({"role": "assistant", "content": clean_sql})
                            messages.append({"role": "user", "content": "The generated SQL doesn't match the schema. Please use the schema tools to verify the correct column names and try again."})
                            continue
                        raise ValueError("Failed to generate valid SQL after validation")
                except Exception as e:
                    current_app.logger.error(f"SQL validation error: {str(e)}")
                    if attempt < max_retries - 1:
                        continue
                    raise
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    current_app.logger.error(f"Failed to generate valid SQL after {max_retries} attempts: {str(e)}")
                    raise
                current_app.logger.warning(f"Error generating SQL, retrying (attempt {attempt + 1}/{max_retries}): {str(e)}")
        
        raise ValueError("Failed to generate valid SQL query")

    
    @staticmethod
    def generate_insight_from_sql_results(sql: str, cols: list, rows: list, question: str = None) -> str:
        """Generate natural language insight from SQL query results.
        
        Args:
            sql: The SQL query
            cols: Column names
            rows: Result rows
            question: Optional question that prompted the SQL query
            
        Returns:
            str: Natural language insight
        """
        client = AIService.get_client()
        
        # Format results as a markdown table
        header = " | ".join(cols)
        separator = " | ".join("---" for _ in cols)
        data_rows = "\n".join(" | ".join(map(str, r)) for r in rows)
        table_md = f"{header}\n{separator}\n{data_rows}"
        
        # Limit the number of rows to reduce token usage
        max_rows = 50
        if len(rows) > max_rows:
            table_md = f"{header}\n{separator}\n" + "\n".join(" | ".join(map(str, r)) for r in rows[:max_rows])
            table_md += f"\n... and {len(rows) - max_rows} more rows"
        
        # Create system message based on whether a question was provided
        if question:
           system_msg = (
                "You are an analytics assistant with access to the results of a SQL query. "
                "Your job is to answer the user's specific question with a short, insight-focused summary. "
                "Always respond in the same language and tone as the **user's original question**: use English if asked in English, use Urdu if asked in Urdu, and use Roman Urdu if asked in Roman Urdu. "
                "Match the user's phrasing and style (e.g., 'car', 'bus', 'aadmi', 'kitnay', 'marnay walay') and keep the response clear and culturally appropriate. "
                "Limit your response to 1–3 short sentences. Do not say things like 'according to the results' or include SQL or metadata — just give the direct insight."
            )

        else:
            system_msg = (
                "You are an analytics assistant with access to the results of a SQL query. Use the SQL and the result table to generate a concise, insight-focused summary. Keep insights brief (max 3 sentences). "
                "Focus on the most significant finding only. "
                "Avoid phrases like 'the data shows' or 'according to the results'."
            )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"SQL Query:\n{sql}\n\nResult Table:\n{table_md}" + (f"\n\nQuestion to answer:\n{question}" if question else "")}
        ]
        
        # Get model based on cost tier
        cost_tier = current_app.config.get('COST_TIER', 'economy')
        model_name = AIService.get_model_by_cost_tier(cost_tier)
        
        # Set token limits for insights
        max_tokens = {
            'gpt-3.5-turbo': 512,   # Economy
            'gpt-4o-mini': 1024,    # Standard
            'gpt-4o': 2048          # Premium
        }.get(model_name, 512)  # Default to economy limit
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens
        )
        
        # Log token usage
        current_app.logger.info(f"Insight token usage - Prompt: {response.usage.prompt_tokens}, "
                             f"Completion: {response.usage.completion_tokens}, "
                             f"Total: {response.usage.total_tokens}")
        
        return response.choices[0].message.content.strip()
    
    @staticmethod
    def generate_insights_from_json(data: str) -> str:
        """Generate concise insights from arbitrary structured JSON data.
        Args:
            data: JSON string or object
        Returns:
            str: Natural-language insights
        """
        client = AIService.get_client()

        # Parse JSON if needed
        try:
            obj = json.loads(data) if isinstance(data, str) else data
        except json.JSONDecodeError as e:
            current_app.logger.warning(f"JSON parse error: {e}, using raw input")
            obj = data

        summary = []

        # Generate summary lines
        summary_lines = AIService.smart_summarize(obj)
        summary_text = "\n".join(summary_lines)

        # Build prompt
        system_msg = (
            "You are a data analyst. Based on the accident summary data, provide:\n"
            "1. One or two clear, quantitative insights (≤30 words each), including exact figures or percentages.\n"
            "   - Focus on accident frequency, types, severity, and timing.\n"
            "   - Include only significant or unusual trends, spikes, or drops.\n"
            "2. Provide one actionable recommendation (≤20 words) ONLY IF a clear, meaningful intervention is supported by the data.\n"
            "   - If no unique or actionable insight is found, DO NOT provide a recommendation.\n"
            "Use precise, concise language suitable for decision-makers. Avoid generic statements."
        )
        user_msg = "Data metrics:\n" + summary_text
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        # Select model and tokens
        tier = current_app.config.get('COST_TIER', 'economy')
        model = AIService.get_model_by_cost_tier(tier)
        max_tokens = {'gpt-3.5-turbo':256, 'gpt-4o-mini':512, 'gpt-4o':1024}.get(model,256)

        # Send to LLM
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens
            )
            usage = resp.usage
            current_app.logger.info(
                f"Tokens - prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, total: {usage.total_tokens}"
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            current_app.logger.error(f"Insight generation error: {e}")
            return f"Error generating insights: {e}"

    @staticmethod
    def _compress_schema(schema: str) -> str:
        """Compress a full SQL dump into a compact schema representation.

        - Extracts CREATE TABLE blocks: table names, columns with types, foreign keys.
        - Extracts all INSERT rows per table, condensing single-value tables into value lists.

        Args:
            schema: Full SQL dump string containing DDL and DML.
        Returns:
            A compressed schema string with tables, columns, types, foreign keys, and sample data.
        """
        if AIService._compressed_schema is not None:
            return AIService._compressed_schema

        text = schema.replace("\r\n", "\n")

        # 1. Extract CREATE TABLE definitions
        create_pattern = re.compile(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w\.]+)\s*\(([^;]+?)\);",
            flags=re.IGNORECASE | re.DOTALL
        )
        create_blocks = create_pattern.findall(text)

        # 2. Extract all INSERT rows per table
        inserts = {}
        insert_pattern = re.compile(
            r"INSERT\s+INTO\s+([\w\.]+)\s*\(([^)]+?)\)\s*VALUES\s*((?:\s*\([^)]+\)\s*,?)+)",
            flags=re.IGNORECASE | re.DOTALL
        )
        for tbl_full, cols_str, vals_group in insert_pattern.findall(text):
            tbl = tbl_full.split('.')[-1]
            cols = [c.strip() for c in cols_str.split(',')]
            row_parts = re.split(r"\)\s*,\s*\(", vals_group.strip().rstrip(','))
            for part in row_parts:
                rt = part.strip().lstrip('(').rstrip(')')
                vals = [v.strip().strip("'\"") for v in rt.split(',')]
                if len(vals) == len(cols):
                    row = dict(zip(cols, vals))
                    inserts.setdefault(tbl, []).append(row)
                elif len(cols) == 1 and vals:
                    inserts.setdefault(tbl, []).append({cols[0]: vals[0]})

        sections = []
        AIService._compressed_schema_hashmap = {}
        
        # 3. Build compressed schema per table
        for tbl_full, cols_block in create_blocks:
            tbl = tbl_full.split('.')[-1]
            columns = {}
            foreign_keys = {}
            
            for line in cols_block.split('\n'):
                line = line.strip().rstrip(',')
                if not line:
                    continue
                    
                # Foreign key detection
                fk_match = re.search(
                    r"FOREIGN\s+KEY\s*\((\w+)\).*?REFERENCES\s+([\w\.]+)\s*\((\w+)\)",
                    line, flags=re.IGNORECASE
                )
                if fk_match:
                    src, ref_full, ref_col = fk_match.groups()
                    ref_tbl = ref_full.split('.')[-1]
                    foreign_keys[src] = f"{ref_tbl}.{ref_col}"
                    continue
                    
                # Skip constraint lines
                if re.match(r"^(PRIMARY|UNIQUE|CONSTRAINT|CHECK|FOREIGN)\b", line, flags=re.IGNORECASE):
                    continue
                    
                # Column definition
                column_pattern = re.compile(
                    r"^\s*(\w+)\s+([a-zA-Z0-9_]+\s*(?:\([^\)]*\))?(?:\s+[A-Z]+)*(?:\s+DEFAULT\s+[^,]+)?(?:\s+NULL|\s+NOT NULL)?)\s*,?",
                    re.IGNORECASE | re.MULTILINE
                )

                m_col = column_pattern.match(line)
                if m_col:
                    name, typ = m_col.groups()
                    columns[name] = typ.strip()

            # Create table schema in required format
            table_schema = {
                "columns": columns,
                "foreign_keys": foreign_keys
            }
            
            # Add sample data if available
            rows = inserts.get(tbl, [])
            if rows:
                table_schema["sample_data"] = rows[:5]  # Limit to 5 sample rows
            
            # Store in hashmap
            AIService._compressed_schema_hashmap[tbl] = table_schema
            
            # Create text representation for sections
            section = [f"Table: {tbl}"]
            section.append("Columns:")
            for col, typ in columns.items():
                section.append(f"  {col}: {typ}")
            if foreign_keys:
                section.append("Foreign Keys:")
                for src, ref in foreign_keys.items():
                    section.append(f"  {src} -> {ref}")
            if rows:
                section.append("Sample Data:")
                for r in rows[:5]:  # Limit to 5 sample rows
                    section.append("  - " + ", ".join(f"{k}={v}" for k, v in r.items()))
            
            sections.append("\n".join(section))

        AIService._compressed_schema = "\n\n".join(sections)
        return AIService._compressed_schema





    @staticmethod
    def list_fine_tuned_models():
        """List all fine-tuned models in the account.
        
        Returns:
            list: List of model information dictionaries
        """
        try:
            client = AIService.get_client()
            jobs = client.fine_tuning.jobs.list()
            
            model_list = []
            for job in jobs:
                if job.fine_tuned_model:  # Only include completed jobs with models
                    # Extract suffix from model ID
                    suffix = None
                    if job.fine_tuned_model:
                        # Model ID format: ft:gpt-3.5-turbo-0125:personal:accident-sql-generator:BXVMbFkQ
                        parts = job.fine_tuned_model.split(':')
                        if len(parts) >= 4:
                            # The last part is usually the unique identifier
                            suffix = parts[-1]
                    
                    model_list.append({
                        'model_id': job.fine_tuned_model,
                        'status': job.status,
                        'created_at': job.created_at,
                        'finished_at': job.finished_at,
                        'base_model': job.model,
                        'training_file': job.training_file,
                        'trained_tokens': job.trained_tokens,
                        'suffix': suffix
                    })
            
            return model_list
        except Exception as e:
            current_app.logger.error(f"Error listing fine-tuned models: {str(e)}")
            return []

    @staticmethod
    def delete_fine_tuned_model(model_id: str) -> bool:
        """Delete a fine-tuned model.
        
        Args:
            model_id: The ID of the model to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            client = AIService.get_client()
            client.models.delete(model_id)
            current_app.logger.info(f"Successfully deleted model: {model_id}")
            return True
        except Exception as e:
            current_app.logger.error(f"Error deleting model {model_id}: {str(e)}")
            return False
        
 
    @staticmethod
    def format_summary_section(key: str, records: list) -> list:
        summary = []

        # Title formatting
        title = key.replace('_', ' ').title().replace(' ', '')
        summary.append(f"== {title} ==")

        # Format hint
        summary.append("(label - count (avg severity))")
        summary.append(f"- Total categories: {len(records)}")

        if all(isinstance(r, dict) for r in records):
            # List each label with count and avg severity
            for r in records:
                label = r.get('label', 'Unknown')
                count = r.get('count', 0)
                avg_severity = r.get('avgSeverity')
                if avg_severity is not None:
                    summary.append(f"{label} - {count:,} ({avg_severity:.2f})")
                else:
                    summary.append(f"{label} - {count:,}")

            # Overall severity range & average
            severities = [r['avgSeverity'] for r in records if r.get('avgSeverity') is not None]
            if severities:
                summary.append(f"- Severity range: {min(severities):.2f} to {max(severities):.2f}")
                summary.append(f"- Overall average severity: {sum(severities)/len(severities):.2f}")

        return summary

    
    from dateutil.parser import parse as parse_date

    @staticmethod
    def detect_trend_frequency(dates: list) -> str:
        if len(dates) < 2:
            return "Unknown"
        try:
            date_objs = [parse_date(d) for d in dates]
            delta = (date_objs[1] - date_objs[0]).days
            if 5 <= delta <= 10:
                return "Weekly"
            elif 25 <= delta <= 35:
                return "Monthly"
            elif 350 <= delta <= 370:
                return "Yearly"
            return "Custom"
        except Exception:
            return "Unknown"

    @staticmethod
    def summarize_trends(records: list) -> list:
        summary = []
        if not records:
            return summary

        # Ensure records are sorted by timePeriod
        records = sorted(records, key=lambda r: r.get("timePeriod", ""))

        # Extract time periods and frequency
        time_periods = [r["timePeriod"] for r in records if "timePeriod" in r]
        frequency = AIService.detect_trend_frequency(time_periods)
        summary.append(f"== {frequency.title()} Trends ==")

        # Time range display
        try:
            start = parse_date(time_periods[0]).strftime("%b %Y")
            end = parse_date(time_periods[-1]).strftime("%b %Y")
        except:
            start = time_periods[0]
            end = time_periods[-1]
        summary.append(f"- Time range: {start} to {end}")

        # Extract incident and fatality counts
        counts = [r.get("totalCount", 0) for r in records]
        fatal_counts = sum(r.get("fatalCount", 0) for r in records)

        if counts:
            summary.append(f"- Incidents: {counts[0]:,} → {counts[-1]:,}")

            # Max and min incidents
            max_idx, max_val = max(enumerate(counts), key=lambda x: x[1])
            min_idx, min_val = min(enumerate(counts), key=lambda x: x[1])
            try:
                max_date = parse_date(time_periods[max_idx]).strftime('%b %Y')
                min_date = parse_date(time_periods[min_idx]).strftime('%b %Y')
            except:
                max_date = time_periods[max_idx]
                min_date = time_periods[min_idx]
            summary.append(f"- Max incidents: {max_val:,} in {max_date}")
            summary.append(f"- Min incidents: {min_val:,} in {min_date}")

            # Month-wise incident breakdown
            summary.append("- Incidents by month:")
            for rec in records:
                try:
                    month = parse_date(rec["timePeriod"]).strftime("%b %Y")
                except:
                    month = rec["timePeriod"]
                summary.append(f"  {month}: {rec.get('totalCount', 0):,}")

        if fatal_counts > 0:
            summary.append(f"- Fatalities: {fatal_counts:,}")

        return summary
    
    @staticmethod
    def smart_summarize(obj: dict) -> list:
        summary = []
        for key, value in obj.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                if key == "trends":
                    summary.extend(AIService.summarize_trends(value))
                else:
                    summary.extend(AIService.format_summary_section(key, value))
            elif isinstance(value, list):
                summary.append(f"{key}: list of {len(value)} values")
            elif isinstance(value, dict):
                summary.append(f"{key}: dict with {len(value)} fields")
            else:
                if isinstance(value, (int, float)):
                    summary.append(f"{key}: {value}")
                elif isinstance(value, str):
                    summary.append(f"{key}: '{value}'")
        print(summary)
        return summary

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """Normalize SQL for comparison by removing semicolons, extra spaces, and standardizing formatting.
        
        Args:
            sql: SQL query string
            
        Returns:
            str: Normalized SQL query
        """
        # Remove semicolons
        sql = sql.replace(';', '')
        
        # Remove extra whitespace and newlines
        sql = ' '.join(sql.split())
        
        # Convert to lowercase
        sql = sql.lower()
        
        # Remove extra spaces around parentheses
        sql = sql.replace('( ', '(').replace(' )', ')')
        
        # Remove extra spaces around commas
        sql = sql.replace(' ,', ',').replace(', ', ',')
        
        # Remove extra spaces around operators
        for op in ['=', '>', '<', '>=', '<=', '<>', '!=', '+', '-', '*', '/']:
            sql = sql.replace(f' {op} ', op).replace(f' {op}', op).replace(f'{op} ', op)
        
        return sql.strip()

    @staticmethod
    def compare_sql_queries_semantically(query1: str, query2: str) -> bool:
        """Compare two SQL queries semantically using the AI model.
        
        Args:
            query1: First SQL query
            query2: Second SQL query
            
        Returns:
            bool: True if queries are semantically equivalent
        """
        try:
                
            client = AIService.get_client()
            
            system_prompt = """
            You are a SQL expert. Compare two SQL queries and determine if they are semantically equivalent. 
            Two queries are semantically equivalent if they would produce the same results, even if written differently. 
            Consider:
            1. Different column orders
            2. Different table aliases
            3. Different JOIN syntax
            4. Different WHERE clause ordering
            5. Different GROUP BY/ORDER BY ordering
            6. Different subquery structures
            7. Different whitespace or formatting
            8. Different semicolon usage

            Ignore ORDER BY clauses **if** they do not affect aggregation or pagination results.
            Ignore aliases if used consistently and don’t change semantics.

            Respond with ONLY 'true' or 'false'.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query 1: {query1}\nQuery 2: {query2}"}
            ]
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            return result == 'true'
            
        except Exception as e:
            current_app.logger.error(f"Error in semantic SQL comparison: {str(e)}")
            return False

    @staticmethod
    def generate_and_compare_sql_batch(questions_data: list, schema_folder: str) -> tuple[list, dict]:
        """Generate SQL for a batch of questions and compare with expected SQL.
        
        Args:
            questions_data: List of dicts with 'question' and 'expectedSql' keys
            schema_folder: Path to schema folder
            
        Returns:
            tuple: (list of results, statistics dict)
        """
        results = []
        model_checked_count = 0
        
        for item in questions_data:
            try:
                # Generate SQL using the existing method
                generated_sql = AIService.generate_sql_from_question(
                    question=item['question'],
                    schema_folder=schema_folder
                )
                
                # Clean both SQLs for comparison
                clean_generated = SQLService._clean_sql_response(generated_sql)
                clean_expected = SQLService._clean_sql_response(item['expectedSql'])
                
                # Initial normalized comparison
                is_match = AIService._normalize_sql(clean_generated) == AIService._normalize_sql(clean_expected)
                checked_by_model = False
                
                # If not a normalized match, try semantic comparison
                if not is_match:
                    checked_by_model = True
                    model_checked_count += 1
                    is_match = AIService.compare_sql_queries_semantically(
                        clean_generated,
                        clean_expected
                    )
                
                # Add results to the item
                item['llmGeneratedQuery'] = generated_sql
                item['match'] = is_match
                item['checkedByModel'] = checked_by_model
                
            except Exception as e:
                current_app.logger.error(f"Error processing question: {str(e)}")
                item['llmGeneratedQuery'] = None
                item['match'] = False
                item['checkedByModel'] = False
                item['error'] = str(e)
            
            results.append(item)
            
        # Calculate statistics
        stats = {
            'total_questions': len(results),
            'matching_queries': sum(1 for r in results if r.get('match', False)),
            'model_checked_queries': model_checked_count,
            'exact_matches': sum(1 for r in results if r.get('match', False) and not r.get('checkedByModel', False)),
            'semantic_matches': sum(1 for r in results if r.get('match', False) and r.get('checkedByModel', False))
        }
        stats['accuracy_percentage'] = (stats['matching_queries'] / stats['total_questions'] * 100) if stats['total_questions'] > 0 else 0
            
        return results, stats
