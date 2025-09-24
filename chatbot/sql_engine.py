import pandas as pd
from sqlalchemy import create_engine, text, inspect
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List, Any, Tuple, Optional
import json
import re
from datetime import datetime, timedelta
import openai

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLEngine:
    """
    SQL Engine Layer for natural language to SQL translation and execution
    """
    
    def __init__(self, db_path: str = 'database/transportation.db', openai_api_key: str = None):
        """
        Initialize SQL Engine
        
        Args:
            db_path: Path to SQLite database
            openai_api_key: OpenAI API key for NL to SQL translation
        """
        self.db_path = db_path
        self.engine = None
        self.schema_info = {}
        self.data_quality_info = {}
        
        # Load OpenAI API key
        if openai_api_key:
            self.openai_api_key = openai_api_key
        else:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found. NL to SQL translation will not be available.")
        else:
            openai.api_key = self.openai_api_key
        
        # Initialize database connection
        self.connect_to_database()
        self.load_schema_info()
        self.assess_data_quality()
        
        # SQL generation prompts
        self.sql_prompt_template = self._initialize_sql_prompt()
    
    def connect_to_database(self):
        """Create database connection"""
        try:
            self.engine = create_engine(f'sqlite:///{self.db_path}')
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def load_schema_info(self):
        """Load database schema information for context"""
        try:
            inspector = inspect(self.engine)
            
            # Get table names
            tables = inspector.get_table_names()
            
            for table in tables:
                columns = inspector.get_columns(table)
                foreign_keys = inspector.get_foreign_keys(table)
                indexes = inspector.get_indexes(table)
                
                self.schema_info[table] = {
                    'columns': [
                        {
                            'name': col['name'],
                            'type': str(col['type']),
                            'nullable': col['nullable'],
                            'default': col['default']
                        }
                        for col in columns
                    ],
                    'foreign_keys': foreign_keys,
                    'indexes': indexes
                }
            
            logger.info(f"Loaded schema information for {len(tables)} tables")
            
        except Exception as e:
            logger.error(f"Error loading schema info: {e}")
    
    def assess_data_quality(self):
        """Assess data quality to inform SQL generation"""
        try:
            with self.engine.connect() as conn:
                # Check CustomerDemographics data quality
                result = conn.execute(text("SELECT COUNT(*) as total, COUNT(Age) as with_age FROM CustomerDemographics"))
                demo_stats = result.fetchone()
                
                # Check TripData date range
                result = conn.execute(text("SELECT MIN(TripDate) as min_date, MAX(TripDate) as max_date, COUNT(*) as total_trips FROM TripData"))
                trip_stats = result.fetchone()
                
                # Check CheckedInUsers
                result = conn.execute(text("SELECT COUNT(*) as total_checkins FROM CheckedInUsers"))
                checkin_stats = result.fetchone()
                
                self.data_quality_info = {
                    'demographics': {
                        'total_users': demo_stats[0],
                        'users_with_age': demo_stats[1],
                        'missing_age_count': demo_stats[0] - demo_stats[1]
                    },
                    'trips': {
                        'total_trips': trip_stats[2],
                        'date_range': {
                            'min': trip_stats[0],
                            'max': trip_stats[1]
                        }
                    },
                    'checkins': {
                        'total_checkins': checkin_stats[0]
                    }
                }
                
                logger.info(f"Data quality assessment completed: {self.data_quality_info}")
                
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            self.data_quality_info = {}
    
    def _initialize_sql_prompt(self) -> str:
        """Initialize the prompt template for SQL generation"""
        return """You are a SQL expert for Fetii's Austin group rideshare database. Convert natural language queries into accurate SQL queries.

Database Schema:
{schema}

Data Quality Information:
{data_quality}

CRITICAL Fetii-Specific Rules:
1. Use proper SQLite syntax with correct table and column names from the actual schema above
2. Always use table aliases for clarity (cd for CustomerDemographics, td for TripData, ci for CheckedInUsers)
3. TripDate is stored as TEXT in 'YYYY-MM-DD HH:MM:SS' format - use DATE() and STRFTIME() functions
4. For temporal queries, check the actual date range in the data before filtering
5. For age filtering: JOIN with CustomerDemographics, but handle NULL ages with proper WHERE clauses
6. For location filtering: Use LIKE with % wildcards, case-insensitive with LOWER()
7. For group size: Use TotalPassengers column directly from TripData
8. For time analysis: Use STRFTIME('%H', TripDate) for hour, STRFTIME('%w', TripDate) for day of week (0=Sunday, 6=Saturday)
9. For Saturday nights: STRFTIME('%w', TripDate) = '6' AND STRFTIME('%H', TripDate) >= '18'
10. Always handle NULL values appropriately with COALESCE or IS NOT NULL checks

Location Matching Patterns for Austin:
- Moody Center: DropOffAddress LIKE '%Moody%' OR DropOffAddress LIKE '%moody%'
- Downtown: DropOffAddress LIKE '%downtown%' OR DropOffAddress LIKE '%6th%' OR DropOffAddress LIKE '%5th%'
- UT Campus: DropOffAddress LIKE '%Campus%' OR DropOffAddress LIKE '%University%'
- Airport: DropOffAddress LIKE '%Airport%'

Time-based Query Guidelines:
- For "last month" queries, use the actual data date range to determine appropriate filtering
- For recent data, use relative date calculations
- For historical data, use absolute date ranges

Example Query Patterns:
- Count queries: SELECT COUNT(*) as count_name FROM table WHERE conditions
- Top/ranking queries: SELECT columns, COUNT(*) as frequency FROM table GROUP BY columns ORDER BY frequency DESC LIMIT N
- Time analysis: SELECT STRFTIME('%H', TripDate) as hour, COUNT(*) FROM TripData GROUP BY hour
- Demographic analysis: Always JOIN with CustomerDemographics and filter out NULL ages when needed

Natural Language Query: {query}

Generate ONLY the SQL query without explanations:"""
    
    def get_schema_context(self) -> str:
        """Generate schema context for SQL prompt"""
        schema_text = "Actual Database Tables and Columns:\n\n"
        
        for table, info in self.schema_info.items():
            if table == 'sqlite_sequence':  # Skip system table
                continue
                
            schema_text += f"{table}:\n"
            for col in info['columns']:
                schema_text += f"  - {col['name']} ({col['type']})\n"
            if info['foreign_keys']:
                schema_text += "  Foreign Keys:\n"
                for fk in info['foreign_keys']:
                    schema_text += f"    - {fk['constrained_columns']} → {fk['referred_table']}.{fk['referred_columns']}\n"
            schema_text += "\n"
        
        return schema_text
    
    def get_data_quality_context(self) -> str:
        """Generate data quality context for SQL prompt"""
        if not self.data_quality_info:
            return "Data quality information not available."
        
        context = "Current Data Quality Status:\n\n"
        
        # Demographics info
        demo = self.data_quality_info.get('demographics', {})
        if demo:
            context += f"CustomerDemographics: {demo.get('total_users', 0)} total users\n"
            context += f"  - Users with age data: {demo.get('users_with_age', 0)}\n"
            context += f"  - Missing age data: {demo.get('missing_age_count', 0)} users\n\n"
        
        # Trip data info
        trips = self.data_quality_info.get('trips', {})
        if trips:
            context += f"TripData: {trips.get('total_trips', 0)} total trips\n"
            date_range = trips.get('date_range', {})
            if date_range:
                context += f"  - Date range: {date_range.get('min', 'Unknown')} to {date_range.get('max', 'Unknown')}\n\n"
        
        # Checkins info
        checkins = self.data_quality_info.get('checkins', {})
        if checkins:
            context += f"CheckedInUsers: {checkins.get('total_checkins', 0)} total check-ins\n\n"
        
        context += "Important Notes:\n"
        context += "- Always filter out NULL ages when age is required for analysis\n"
        context += "- Use actual date ranges for temporal filtering\n"
        context += "- Consider data completeness when interpreting results\n"
        
        return context
    
    def translate_to_sql(self, natural_query: str) -> Dict[str, Any]:
        """
        Translate natural language query to SQL
        
        Args:
            natural_query: User's natural language query
            
        Returns:
            Dictionary with SQL query and metadata
        """
        if not self.openai_api_key:
            return {
                'success': False,
                'error': 'OpenAI API key not available',
                'sql_query': None
            }
        
        try:
            # Prepare schema context
            schema_context = self.get_schema_context()
            data_quality_context = self.get_data_quality_context()
            
            # Format prompt
            prompt = self.sql_prompt_template.format(
                schema=schema_context,
                data_quality=data_quality_context,
                query=natural_query
            )
            
            # Call OpenAI API
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Generate clean, executable SQL queries that handle NULL values and use proper SQLite syntax."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.1
            )
            
            # Extract SQL from response
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the SQL (remove markdown formatting if present)
            sql_query = re.sub(r'^```sql\s*', '', sql_query)
            sql_query = re.sub(r'\s*```$', '', sql_query)
            sql_query = sql_query.strip()
            
            # Remove any trailing semicolon for consistency
            sql_query = sql_query.rstrip(';')
            
            return {
                'success': True,
                'sql_query': sql_query,
                'natural_query': natural_query,
                'method': 'openai_translation'
            }
            
        except Exception as e:
            logger.error(f"Error translating to SQL: {e}")
            return {
                'success': False,
                'error': str(e),
                'sql_query': None
            }
    
    def validate_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL query without executing it
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Basic syntax validation
            sql_query = sql_query.strip()
            
            if not sql_query:
                return {'valid': False, 'error': 'Empty SQL query'}
            
            # Check for dangerous operations
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
            query_upper = sql_query.upper()
            
            for keyword in dangerous_keywords:
                if keyword in query_upper and not query_upper.startswith('SELECT'):
                    return {
                        'valid': False, 
                        'error': f'Potentially dangerous operation: {keyword}'
                    }
            
            # Try to parse the query (dry run)
            with self.engine.connect() as conn:
                # Use EXPLAIN to validate without executing
                explain_query = f"EXPLAIN QUERY PLAN {sql_query}"
                conn.execute(text(explain_query))
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def execute_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute SQL query and return results
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Dictionary with query results and metadata
        """
        try:
            # Validate query first
            validation = self.validate_sql_query(sql_query)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': f"Invalid SQL: {validation['error']}",
                    'data': None,
                    'sql_query': sql_query
                }
            
            # Execute query
            start_time = datetime.now()
            
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                
                # Handle different types of results
                if result.returns_rows:
                    # SELECT query - return data as DataFrame
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return {
                        'success': True,
                        'data': df,
                        'row_count': len(df),
                        'columns': list(df.columns),
                        'execution_time': execution_time,
                        'sql_query': sql_query,
                        'data_type': 'dataframe'
                    }
                else:
                    # Non-SELECT query (shouldn't happen with our validation, but just in case)
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return {
                        'success': True,
                        'data': {'message': 'Query executed successfully'},
                        'row_count': 0,
                        'execution_time': execution_time,
                        'sql_query': sql_query,
                        'data_type': 'message'
                    }
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None,
                'sql_query': sql_query
            }
    
    def process_natural_language_query(self, natural_query: str) -> Dict[str, Any]:
        """
        Complete pipeline: NL → SQL → Execute → Results
        
        Args:
            natural_query: User's natural language query
            
        Returns:
            Dictionary with complete processing results
        """
        try:
            # Step 1: Translate to SQL
            translation_result = self.translate_to_sql(natural_query)
            
            if not translation_result['success']:
                return {
                    'success': False,
                    'error': translation_result['error'],
                    'stage': 'translation',
                    'natural_query': natural_query
                }
            
            sql_query = translation_result['sql_query']
            
            # Step 2: Execute SQL
            execution_result = self.execute_sql_query(sql_query)
            
            # Combine results
            result = {
                'success': execution_result['success'],
                'natural_query': natural_query,
                'sql_query': sql_query,
                'translation_method': translation_result['method'],
                'stage': 'execution' if execution_result['success'] else 'execution_failed'
            }
            
            if execution_result['success']:
                result.update({
                    'data': execution_result['data'],
                    'row_count': execution_result['row_count'],
                    'columns': execution_result.get('columns', []),
                    'execution_time': execution_result['execution_time'],
                    'data_type': execution_result['data_type']
                })
                
                # Add data quality notes
                result['data_quality_notes'] = self._generate_data_quality_notes(execution_result['data'])
            else:
                result.update({
                    'error': execution_result['error'],
                    'data': None
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in natural language query processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'stage': 'pipeline_error',
                'natural_query': natural_query
            }
    
    def _generate_data_quality_notes(self, data) -> str:
        """Generate data quality notes for the results"""
        notes = []
        
        try:
            if hasattr(data, 'empty') and data.empty:
                notes.append("Query returned no results")
            elif hasattr(data, '__len__') and len(data) == 0:
                notes.append("Query returned empty dataset")
            else:
                # Check for null values in results
                if hasattr(data, 'isnull'):
                    null_counts = data.isnull().sum()
                    if null_counts.any():
                        null_cols = [col for col, count in null_counts.items() if count > 0]
                        notes.append(f"Some null values found in: {', '.join(null_cols)}")
                
                # Check for age-related queries
                if hasattr(data, 'columns') and any('age' in col.lower() for col in data.columns):
                    demo_info = self.data_quality_info.get('demographics', {})
                    missing_age = demo_info.get('missing_age_count', 0)
                    if missing_age > 0:
                        notes.append(f"Note: {missing_age} users have missing age data")
        
        except Exception as e:
            logger.warning(f"Error generating data quality notes: {e}")
            notes.append("Data quality assessment unavailable")
        
        return "; ".join(notes) if notes else "Data appears complete"
    
    def get_sample_queries(self) -> List[Dict[str, str]]:
        """Return sample queries for testing"""
        return [
            {
                'natural': 'How many groups went to Moody Center last month?',
                'description': 'Count trips to Moody Center with temporal filtering'
            },
            {
                'natural': 'What are the top drop-off spots for 18-24 year-olds on Saturday nights?',
                'description': 'Demographic and temporal analysis with location ranking'
            },
            {
                'natural': 'When do large groups (6+ riders) typically ride downtown?',
                'description': 'Time analysis for large groups with location filtering'
            },
            {
                'natural': 'What is the average group size for Fetii rides?',
                'description': 'Simple aggregation query'
            },
            {
                'natural': 'How many users are in the database?',
                'description': 'Basic count query'
            },
            {
                'natural': 'Show me trips with more than 8 passengers',
                'description': 'Filtering by group size'
            }
        ]

def main():
    """Test the SQL Engine functionality"""
    print("Testing SQL Engine")
    print("=" * 60)
    
    # Initialize engine
    try:
        engine = SQLEngine()
    except Exception as e:
        print(f"Failed to initialize SQL Engine: {e}")
        return
    
    # Display data quality info
    print("Data Quality Assessment:")
    print("-" * 40)
    print(engine.get_data_quality_context())
    
    # Test sample queries
    sample_queries = engine.get_sample_queries()
    
    for i, query_info in enumerate(sample_queries[:3], 1):  # Test first 3
        print(f"\n{i}. Testing: '{query_info['natural']}'")
        print(f"   Description: {query_info['description']}")
        print("-" * 40)
        
        result = engine.process_natural_language_query(query_info['natural'])
        
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Generated SQL: {result['sql_query']}")
            print(f"Rows returned: {result['row_count']}")
            print(f"Data quality: {result.get('data_quality_notes', 'N/A')}")
            if result['row_count'] > 0 and hasattr(result['data'], 'head'):
                print(f"Sample data:\n{result['data'].head(2)}")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()
