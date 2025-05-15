import os
from openai import OpenAI
from flask import current_app
import sys
import os.path
import json
import random
import time

# Add the parent directory of app to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.services.sql_service import SQLService
system_prompt_global = (
            "You are a PostgreSQL SQL assistant for a traffic accident reporting system. "
            "Generate only valid SQL SELECT statements based on user questions. "
            "Use correct table and column names. Do not include explanations or comments."
        )

class AIService:
    """Service for interacting with OpenAI API."""
    
    # Class variable to cache the schema
    _schema_cache = {}
    
    @staticmethod
    def get_client():
        """Get OpenAI client with API key from environment."""
        try:
            api_key = current_app.config.get('OPENAI_API_KEY')
            if not api_key:
                current_app.logger.error("OpenAI API key is missing from configuration")
                raise ValueError("OpenAI API key not configured")
            return OpenAI(api_key=api_key)
        except Exception as e:
            current_app.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    def generate_fine_tuning_data_batch(self, num_examples=15, output_file=None, compress_schema=True):
        """Generate fine-tuning data in JSONL format with batched SQL generation."""
        try:
            schema_folder = current_app.config.get('SCHEMA_FOLDER')
            current_app.logger.info(f"Generating fine-tuning data from schema in {schema_folder}")

            if not output_file:
                timestamp = int(time.time())
                output_file = os.path.join(schema_folder, f"fine_tuning_data_{timestamp}.jsonl")

            # Load and compress schema
            schema = self.get_schema(schema_folder)
            if compress_schema:
                current_app.logger.info("Compressing schema to reduce token usage")
                schema = self._compress_schema(schema)

            sample_queries = [
                "Show all severe accidents in Karachi from last month",
                "How many accidents involved motorcycles in Lahore?",
                "Count the number of accidents by type in Islamabad",
                "Find accidents with more than 3 casualties",
                "List all accidents where ambulances were dispatched",
                "Show accidents with hospital transfers in the last week",
                "Find average response time for ambulances in Karachi",
                "List all accidents involving pedestrians",
                "Show accidents with driver negligence as the cause",
                "Count accidents by weather condition",
                "What areas have the highest number of accidents?",
                "Show accidents that occurred during rainy weather",
                "Which vehicle type is involved in the most accidents?",
                "List all dispatches that took more than 30 minutes",
                "Show accidents on highways with poor visibility"
            ]

            # Add variations
            all_queries = []
            for q in sample_queries:
                all_queries += [
                    q,
                    q.replace("Show", "List"),
                    q.replace("Find", "Get"),
                    f"I need to {q.lower()}",
                    f"Could you {q.lower()}?"
                ]

            if len(all_queries) > num_examples:
                all_queries = random.sample(all_queries, num_examples)
            elif len(all_queries) < num_examples:
                while len(all_queries) < num_examples:
                    base = random.choice(sample_queries)
                    all_queries.append(f"Please tell me about {base.lower()}")
 
            # Batch in groups of 5
            batch_size = 5
            current_app.logger.info(f"Starting batch of size: {batch_size}")
            all_examples = []
            for i in range(0, len(all_queries), batch_size):
                batch_queries = all_queries[i:i + batch_size]
                examples = self._generate_batch_sql_examples(batch_queries, schema)
                all_examples.extend(examples)

            with open(output_file, 'w') as f:
                for ex in all_examples:
                    f.write(json.dumps(ex) + '\n')

            current_app.logger.info(f"Saved {len(all_examples)} examples to {output_file}")
            return output_file

        except Exception as e:
            current_app.logger.error(f"Error generating fine-tuning data: {str(e)}")
            return None

    def _generate_batch_sql_examples(self, queries, schema):
        """Generate SQL queries for a batch of natural language questions."""
        system_prompt = (
            "You are a PostgreSQL SQL assistant for a traffic accident reporting system. "
            "Generate only valid SQL SELECT statements based on user questions. "
            "Use correct table and column names. Do not include explanations or comments."
        )

        prompt = "Generate SQL queries for the following questions:\n\n"
        for idx, q in enumerate(queries):
            prompt += f"Question {idx+1}: {q}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Schema:\n{schema}"},
            {"role": "user", "content": prompt}
        ]

        client = self.get_client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=2048
        )

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
            extracted_sql.append(sql)

        # Pad if missing
        while len(extracted_sql) < len(queries):
            extracted_sql.append("SELECT * FROM accident_reports LIMIT 10;")

        examples = []
        for user_msg, sql_msg in zip(queries, extracted_sql):
            examples.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": sql_msg}
                ]
            })

        return examples

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
    def get_schema(schema_folder: str) -> str:
        """Get database schema, using cache if available.
        
        Args:
            schema_folder: Path to folder containing schema definitions
            
        Returns:
            str: Combined schema and sample data information
        """
        # Check if schema is already in cache
        if schema_folder in AIService._schema_cache:
            current_app.logger.info(f"Using cached schema from {schema_folder}")
            return AIService._schema_cache[schema_folder]
        
        # If not in cache, load it and store in cache
        current_app.logger.info(f"Loading schema from {schema_folder}")
        schema = SQLService.load_schema_from_folder(schema_folder)
        AIService._schema_cache[schema_folder] = schema
        return schema
    
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
    
    def generate_fine_tuning_data(self, num_examples=15, output_file=None, compress_schema=True):
        """Generate fine-tuning data to create a schema-aware model.
        
        Args:
            num_examples: Number of examples to generate
            output_file: Path to save fine-tuning data (generated if not provided)
            compress_schema: Whether to compress the schema to reduce tokens
            
        Returns:
            str: Path to the output file if successful, None otherwise
        """
        try:
            schema_folder = current_app.config.get('SCHEMA_FOLDER')
            current_app.logger.info(f"Generating fine-tuning data from schema in {schema_folder}")
            
            # Generate default output file if not provided
            if not output_file:
                timestamp = int(time.time())
                output_file = f"fine_tuning_data_{timestamp}.jsonl"
                # If schema folder is set, save there
                if schema_folder:
                    output_file = os.path.join(schema_folder, output_file)
            
            # Load schema
            schema = self.get_schema(schema_folder)
            
            # Compress schema if requested to reduce token usage
            if compress_schema:
                current_app.logger.info("Compressing schema to reduce token usage")
                schema = self._compress_schema(schema)
            
            # Sample queries to use as examples
            sample_queries = [
                "Show all severe accidents in Karachi from last month",
                "How many accidents involved motorcycles in Lahore?",
                "Count the number of accidents by type in Islamabad",
                "Find accidents with more than 3 casualties",
                "List all accidents where ambulances were dispatched",
                "Show accidents with hospital transfers in the last week",
                "Find average response time for ambulances in Karachi",
                "List all accidents involving pedestrians",
                "Show accidents with driver negligence as the cause",
                "Count accidents by weather condition",
                "What areas have the highest number of accidents?",
                "Show accidents that occurred during rainy weather",
                "Which vehicle type is involved in the most accidents?",
                "List all dispatches that took more than 30 minutes",
                "Show accidents on highways with poor visibility"
            ]
            
            # Generate query variations
            all_queries = []
            for query in sample_queries:
                all_queries.append(query)
                # Add variations of each query
                all_queries.append(query.replace("Show", "List"))
                all_queries.append(query.replace("Find", "Get"))
                all_queries.append(f"I need to {query.lower()}")
                all_queries.append(f"Could you {query.lower()}?")
            
            # Select random subset if we have more than needed
            if len(all_queries) > num_examples:
                all_queries = random.sample(all_queries, num_examples)
            elif len(all_queries) < num_examples:
                # If we need more examples, duplicate and modify some
                while len(all_queries) < num_examples:
                    base_query = random.choice(sample_queries)
                    variation = f"Please tell me about {base_query.lower().replace('show', '').replace('find', '').replace('list', '')}"
                    all_queries.append(variation)
            
            # =================== OPTIMIZED BATCH APPROACH =====================
            current_app.logger.info(f"Generating SQL for {len(all_queries)} queries using batch approach")
            
            # Generate fine-tuning examples
            examples = []
            
            # Create a single batch request for all queries
            batch_prompt = "Generate SQL queries for the following questions. For each question, output only the SQL query without any explanations or formatting:\n\n"
            for i, query in enumerate(all_queries):
                batch_prompt += f"Question {i+1}: {query}\n"
            
            # Make a single API call for all examples
            model_name = "gpt-3.5-turbo-1106"  # Use a more cost-effective model for batch generation
            client = self.get_client()
            system_prompt = (
            "You are a PostgreSQL SQL assistant for a traffic accident reporting system. "
            "Generate only valid SQL SELECT statements based on user questions. "
            "Use correct table and column names. Do not include explanations or comments."
        )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Database schema definitions and some lovs data:\n{schema}"},
                {"role": "user", "content": batch_prompt}
            ]
            
            current_app.logger.info("Sending batch request to generate SQL queries")
            token_estimate = len(schema) // 4
            current_app.logger.info(f"Estimated schema tokens: ~{token_estimate} (sent only once)")
            
            # Calculate max tokens to avoid hitting context limits
            max_tokens_per_query = 200  # Reasonable estimate for SQL query length
            max_tokens = len(all_queries) * max_tokens_per_query
            current_app.logger.info(f"Setting max_tokens={max_tokens} for batch response")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens
            )
            
            # Parse the response to extract individual SQL queries
            response_text = response.choices[0].message.content.strip()
            current_app.logger.info("Received batch response, parsing SQL queries")
            
            # Split by SQL query markers
            sql_blocks = response_text.split("SQL ")
            
            # Process each SQL query
            extracted_queries = []
            for block in sql_blocks:
                if not block.strip():
                    continue
                    
                # Extract the query part after the number and colon
                parts = block.split(":", 1)
                if len(parts) < 2:
                    continue
                    
                sql = parts[1].strip()
                extracted_queries.append(sql)
            
            # Match SQL queries back to original questions
            current_app.logger.info(f"Extracted {len(extracted_queries)} SQL queries from response")
            
            # Ensure we have enough queries
            if len(extracted_queries) < len(all_queries):
                current_app.logger.warning(f"Only extracted {len(extracted_queries)} SQL queries for {len(all_queries)} questions")
                # Pad with placeholders if needed
                while len(extracted_queries) < len(all_queries):
                    extracted_queries.append("SELECT * FROM accident_reports LIMIT 10")
            
            # Create the training examples
            for i, (query, sql) in enumerate(zip(all_queries, extracted_queries)):
                # Create fine-tuning example
                example = {
                    "messages": [
                        {"role": "system", "content": "You are an assistant that generates SQL queries for a Traffic Accident Reporting System. ONLY output a single valid SQL SELECT statement, and nothing else."},
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
            job = client.fine_tuning.jobs.create(
                training_file=training_file.id,
                model="gpt-3.5-turbo",  # Base model to fine-tune
                suffix=suffix
            )
            
            current_app.logger.info(f"Created fine-tuning job: {job.id}")
            
            # Check job status (optional - this can take a while)
            current_app.logger.info("Checking initial job status...")
            job_status = client.fine_tuning.jobs.retrieve(job.id)
            current_app.logger.info(f"Job status: {job_status.status}")
            
            # Get the model ID from the job
            # The model ID will be in format "ft:gpt-3.5-turbo-0125:[org]:custom:[suffix]-[timestamp]"
            model_id = f"ft:{job_status.model if job_status.model else 'gpt-3.5-turbo'}"
            
            # Important: Note that the fine-tuning job will continue running in the 
            # background and may take 1-4 hours to complete
            current_app.logger.info(f"Model ID will be: {model_id}")
            current_app.logger.info("Fine-tuning job is running and may take 1-4 hours to complete")
            
            return model_id
            
        except Exception as e:
            current_app.logger.error(f"Error creating fine-tuned model: {str(e)}")
            return None
    
    @staticmethod
    def wait_for_fine_tuning(job_id, timeout_minutes=60, check_interval=60):
        """Wait for fine-tuning job to complete.
        
        Args:
            job_id: The fine-tuning job ID
            timeout_minutes: Maximum time to wait in minutes
            check_interval: Seconds between status checks
            
        Returns:
            tuple: (success, model_id or error message)
        """
        try:
            client = AIService.get_client()
            start_time = time.time()
            end_time = start_time + (timeout_minutes * 60)
            
            current_app.logger.info(f"Waiting for fine-tuning job {job_id} to complete (timeout: {timeout_minutes} min)")
            
            while time.time() < end_time:
                job = client.fine_tuning.jobs.retrieve(job_id)
                status = job.status
                
                if status == "succeeded":
                    model_id = job.fine_tuned_model
                    current_app.logger.info(f"Fine-tuning job completed successfully. Model ID: {model_id}")
                    return True, model_id
                elif status in ["failed", "cancelled"]:
                    error_msg = f"Fine-tuning job failed with status: {status}"
                    current_app.logger.error(error_msg)
                    return False, error_msg
                
                current_app.logger.info(f"Job status: {status}. Checking again in {check_interval} seconds...")
                time.sleep(check_interval)
            
            return False, f"Timeout waiting for fine-tuning job to complete after {timeout_minutes} minutes"
            
        except Exception as e:
            error_msg = f"Error waiting for fine-tuning job: {str(e)}"
            current_app.logger.error(error_msg)
            return False, error_msg
    
    @staticmethod
    def generate_sql_from_question(question: str, schema_folder: str) -> str:
        """Generate SQL query from natural language question.
        
        Args:
            question: The natural language question
            schema_folder: Path to folder containing schema definitions
            
        Returns:
            str: Generated SQL query
        """
        client = AIService.get_client()
        
        # Get the cost tier from configuration
        cost_tier = current_app.config.get('COST_TIER', 'economy')
        
        # Check if we're using a pre-trained model that already knows the schema
        use_schema_aware_model = current_app.config.get('USE_SCHEMA_AWARE_MODEL', True)
        
        # Get the appropriate model based on cost tier (can be overridden by SQL_MODEL config)
        model_name = current_app.config.get('SQL_MODEL', AIService.get_model_by_cost_tier(cost_tier))
                # at top of your request logic

        available = client.models.list().data
        if not any(m.id == model_name for m in available):
            raise ValueError(f"Model {model_name!r} not found in your account")
        # Log the model being used for this query
        current_app.logger.info(f"Using model: {model_name} (cost tier: {cost_tier})")
        
        if use_schema_aware_model:
            # When using a model that's already trained on our schema, we don't need to send it again
            current_app.logger.info(f"Using schema-aware model: {model_name}")
            messages = [
                {"role": "system", "content": system_prompt_global},
                {"role": "user", "content": question}
            ]
        else:
            # Standard approach: load schema and include it with the request
            current_app.logger.info(f"Including schema with request to model: {model_name}")
            schema = AIService.get_schema(schema_folder)
            messages = [
                {"role": "system", "content": system_prompt_global},
                {"role": "system", "content": f"Database schema definitions:\n{schema}"},
                {"role": "user", "content": question}
            ]
        
        # Log token usage estimation based on approach
        if not use_schema_aware_model:
            schema_size = len(AIService.get_schema(schema_folder))
            current_app.logger.info(f"Estimated schema tokens: ~{schema_size // 4} tokens")
        
        # Adjust temperature based on model - higher-quality models can use lower temperature
        temperature = 0.0 if 'gpt-4' in model_name else 0.2
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        
        # Get the response content
        raw_sql = response.choices[0].message.content.strip()
        
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
        
        return clean_sql
    
    @staticmethod
    def generate_insight_from_sql_results(sql: str, cols: list, rows: list) -> str:
        """Generate natural language insight from SQL query results.
        
        Args:
            sql: The SQL query
            cols: Column names
            rows: Result rows
            
        Returns:
            str: Natural language insight
        """
        client = AIService.get_client()
        
        # Format results as a markdown table
        header = " | ".join(cols)
        separator = " | ".join("---" for _ in cols)
        data_rows = "\n".join(" | ".join(map(str, r)) for r in rows)
        table_md = f"{header}\n{separator}\n{data_rows}"
        
        messages = [
            {"role": "system", "content": "You are an analytics assistant. Keep insights brief (max 3 sentences). Focus on the most significant finding only. Avoid phrases like 'the data shows' or 'according to the results'."},
            {"role": "user", "content": f"SQL Query:\n{sql}\n\nResult Table:\n{table_md}"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
    
    @staticmethod
    def generate_insights_from_json(data: str) -> str:
        """Generate insights from JSON data.
        
        Args:
            data: str containing json data
            
        Returns:
            str: Natural language insights about the data
        """
        client = AIService.get_client()
        
        try:
            # Parse the JSON string
            try:
                json_data = json.loads(data) if isinstance(data, str) else data
            except json.JSONDecodeError as e:
                json_data = data
                current_app.logger.warning(f"Invalid JSON data: {str(e)}")
            
            # Build a concise statistical/textual summary
            if isinstance(json_data, list):
                data_summary = f"Dataset contains {len(json_data)} records."
                if len(json_data) > 0 and isinstance(json_data[0], dict):
                    fields = list(json_data[0].keys())
                    data_summary += f" Fields: {', '.join(fields)}."
                    for field in fields:
                        vals = [item.get(field) for item in json_data if isinstance(item.get(field), (int, float))]
                        if vals:
                            avg = sum(vals) / len(vals)
                            data_summary += f" Average {field}: {avg:.2f}."
            elif isinstance(json_data, dict):
                data_summary = f"Data contains keys: {', '.join(json_data.keys())}."
            else:
                data_summary = "Non-structured data received."
            
            # Prepare prompt
            messages = [
                {"role": "system", "content": (
                    "You are a concise data analyst. Analyze the traffic accident data and provide:\n"
                    "1. THREE key points (1-2 sentences each)\n"
                    "2. ONE recommendation (1 sentence)\n\n"
                    "Example format:\n"
                    "1. [First key point]\n"
                    "2. [Second key point]\n"
                    "3. [Third key point]\n\n"
                    "Keep total response under 150 words."
                )},
                {"role": "user", "content": f"Data summary: {data_summary}"}
            ]
            
            # Call the chat model directly
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            current_app.logger.error(f"Error generating insights: {str(e)}")
            return f"Error generating insights: {str(e)}"
    
    @staticmethod
    def ask_database_with_insight(question: str, schema_folder: str):
        """Ask natural language question and get SQL, data, and insights.
        
        Args:
            question: Natural language question
            schema_folder: Path to folder with schema definitions
            
        Returns:
            dict: Result with SQL, columns, rows, and insight
        """
        try:
            # Generate SQL query
            sql = AIService.generate_sql_from_question(question, schema_folder)
            
            # Validate the SQL query
            if not SQLService.validate_select(sql):
                raise ValueError(f"Invalid SQL generated: {sql}")
                
            # Execute the SQL query
            cols, rows = SQLService.run_sql(sql)
            
            # Generate insight from the results
            insight = AIService.generate_insight_from_sql_results(sql, cols, rows)
            
            return {
                "sql": sql,
                "columns": cols,
                "rows": rows,
                "insight": insight
            }
        except Exception as e:
            current_app.logger.error(f"Error in ask_database_with_insight: {str(e)}")
            # Re-raise the exception to be handled by the route error handler
            raise

    @staticmethod
    def generate_insights(prompt, data):
        """Generate insights from data using OpenAI."""
        client = AIService.get_client()
        
        # Prepare context with data for the model
        context = f"Data about incidents: {str(data)}\n\n"
        full_prompt = context + prompt
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use appropriate model
            messages=[
                {"role": "system", "content": """You are a concise incident analyst. Analyze the data and provide:
1. THREE key points (1-2 sentences each)
2. ONE recommendation (1 sentence)

Example format:
1. [First key point about patterns in 1-2 sentences]
2. [Second key point about trends in 1-2 sentences] 
3. [Third key point about notable findings in 1-2 sentences]

Keep total response under 150 words."""},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.1,  # Lower temperature for more factual responses
            max_tokens=300
        )
        
        # Extract and return the generated content
        return response.choices[0].message.content
        
    @staticmethod
    def analyze_area_incidents(area, incidents):
        """Analyze incidents in a specific area."""
        incidents_in_area = [i for i in incidents if i.area.lower() == area.lower()]
        
        if not incidents_in_area:
            return f"No incidents found in {area}."
        
        prompt = f"Analyze the following incidents in {area} and provide key insights about patterns, frequency, and severity:"
        
        return AIService.generate_insights(prompt, incidents_in_area)

    @staticmethod
    def _compress_schema(schema):
        """Compress schema to reduce token usage.
        This creates a more compact representation focusing on table names and columns.
        
        Args:
            schema: Original schema string
            
        Returns:
            str: Compressed schema
        """
        try:
            # Extract tables and their columns from the schema
            tables = []
            lines = schema.split('\n')
            current_table = None
            
            for line in lines:
                line = line.strip()
                # Look for CREATE TABLE statements
                if line.startswith('CREATE TABLE') and '(' in line:
                    table_name = line.split('CREATE TABLE')[1].split('(')[0].strip()
                    current_table = {'name': table_name, 'columns': []}
                    tables.append(current_table)
                
                # Look for column definitions when we're inside a table
                elif current_table and line and line[0].isalnum() and ' ' in line:
                    # Simple parsing for column definitions
                    col_parts = line.split(' ', 1)
                    if len(col_parts) == 2:
                        col_name = col_parts[0].replace(',', '').strip()
                        col_type = col_parts[1].replace(',', '').strip()
                        if col_name and not col_name.upper() in ['PRIMARY', 'FOREIGN', 'CONSTRAINT', 'CHECK']:
                            current_table['columns'].append(f"{col_name}({col_type})")
                
                # End of table definition
                elif line.startswith(');'):
                    current_table = None
            
            # Extract sample data patterns from LOV DATA sections
            lov_data = []
            in_lov = False
            lov_section = ""
            
            for line in lines:
                if "-- LOV DATA" in line:
                    in_lov = True
                    lov_section = line.strip() + "\n"
                elif in_lov and "-- END LOV DATA" in line:
                    lov_section += line.strip()
                    lov_data.append(lov_section)
                    in_lov = False
                elif in_lov:
                    lov_section += line.strip() + "\n"
            
            # Build compressed schema
            compressed = "-- COMPRESSED SCHEMA FOR TOKEN EFFICIENCY\n\n"
            
            # Add tables in compact form
            compressed += "-- TABLES\n"
            for table in tables:
                compressed += f"• {table['name']}({', '.join(table['columns'])})\n"
            
            # Add key sample data patterns (limit to 3)
            if lov_data:
                compressed += "\n-- SAMPLE DATA PATTERNS\n"
                for i, lov in enumerate(lov_data[:3]):
                    compressed += f"{lov}\n"
            
            return compressed
            
        except Exception as e:
            current_app.logger.warning(f"Schema compression failed, using original schema: {str(e)}")
            return schema 