import pandas as pd
from sqlalchemy import create_engine, text, inspect
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List, Any, Tuple, Optional
import json
import re
from datetime import datetime
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
    
    def _initialize_sql_prompt(self) -> str:
        """Initialize the prompt template for SQL generation"""
        return """You are a SQL expert for Fetii's Austin rideshare database. Generate EXACT SQL queries that return precise data.

CRITICAL REQUIREMENTS:
1. Use EXACT column names and table names as provided in schema
2. For location queries, use LIKE with wildcards: WHERE column LIKE '%keyword%'
3. For date queries, use DATE() function for SQLite: WHERE DATE(TripDate) = 'YYYY-MM-DD'
4. For group size queries, use TotalPassengers column from TripData table
5. For age queries, use Age column from CustomerDemographics table
6. Always JOIN tables properly when needed
7. Count records with COUNT(*), not estimations
8. Use DISTINCT when counting unique entities

Database Schema:
{schema}

Fetii-Specific Rules:
1. Use proper SQLite syntax
2. Always use table aliases for clarity
3. Use proper JOIN syntax when multiple tables are needed
4. Include appropriate WHERE clauses for filtering
5. Use aggregate functions (COUNT, SUM, AVG, etc.) when needed
6. For date filtering: TripDate is stored as TEXT in format 'MM/DD/YY HH:MM'
7. For time analysis: Use strftime('%H', TripDate) for hour, strftime('%w', TripDate) for day of week
8. For age filtering: JOIN with CustomerDemographics and filter by Age ranges
9. For location filtering: Use LIKE with % wildcards for partial matches
10. For group size: Use TotalPassengers column for group size analysis

Fetii Query Examples:
- "How many groups went to Moody Center last month?" → 
  SELECT COUNT(*) as moody_trips 
  FROM TripData 
  WHERE DropOffAddress LIKE '%Moody%' 
    AND DATE(TripDate) >= DATE('now', '-1 month')
- "Top drop-off spots for 18-24 year-olds on Saturday nights?" → 
  SELECT t.DropOffAddress, COUNT(*) as trip_count
  FROM TripData t
  JOIN CheckedInUsers ci ON t.TripID = ci.TripID
  JOIN CustomerDemographics cd ON ci.UserID = cd.UserID
  WHERE cd.Age BETWEEN 18 AND 24 
    AND strftime('%w', t.TripDate) = '6' 
    AND strftime('%H', t.TripDate) >= '18'
  GROUP BY t.DropOffAddress
  ORDER BY trip_count DESC
  LIMIT 10
- "When do large groups (6+ riders) typically ride downtown?" → 
  SELECT strftime('%H', TripDate) as hour, COUNT(*) as trip_count
  FROM TripData 
  WHERE TotalPassengers >= 6 
    AND (DropOffAddress LIKE '%downtown%' OR DropOffAddress LIKE '%6th%')
  GROUP BY hour
  ORDER BY trip_count DESC
- "What's the average group size for Fetii rides in Austin?" →
  SELECT AVG(TotalPassengers) as avg_group_size, 
         COUNT(*) as total_trips,
         MIN(TotalPassengers) as min_group_size,
         MAX(TotalPassengers) as max_group_size
  FROM TripData
- "Which Austin neighborhoods have the most Fetii activity?" →
  SELECT 
    CASE 
      WHEN DropOffAddress LIKE '%downtown%' OR DropOffAddress LIKE '%6th%' THEN 'Downtown'
      WHEN DropOffAddress LIKE '%moody%' THEN 'Moody Center Area'
      WHEN DropOffAddress LIKE '%university%' OR DropOffAddress LIKE '%campus%' THEN 'University Area'
      WHEN DropOffAddress LIKE '%south%' THEN 'South Austin'
      WHEN DropOffAddress LIKE '%north%' THEN 'North Austin'
      ELSE 'Other'
    END as neighborhood,
    COUNT(*) as trip_count
  FROM TripData
  GROUP BY neighborhood
  ORDER BY trip_count DESC

Natural Language Query: {query}

SQL Query:"""
    
    def get_schema_context(self) -> str:
        """Generate schema context for SQL prompt"""
        schema_text = "Tables and Columns:\n\n"
        
        for table, info in self.schema_info.items():
            schema_text += f"{table}:\n"
            for col in info['columns']:
                schema_text += f"  - {col['name']} ({col['type']})\n"
            if info['foreign_keys']:
                schema_text += "  Foreign Keys:\n"
                for fk in info['foreign_keys']:
                    schema_text += f"    - {fk['constrained_columns']} → {fk['referred_table']}.{fk['referred_columns']}\n"
            schema_text += "\n"

        # Derived analytics views to improve query accuracy
        schema_text += (
            "Derived Views Available:\n"
            "- TripFeatures(TripID, BookingUserID, PickUpAddress, DropOffAddress, TotalPassengers, TripDateTime, TripDate, TripYearMonth, TripHour, TripWeekday, IsWeekend, PickUpLabel, DropOffLabel)\n"
            "- TripPassengerAges(TripID, UserID, Age)\n"
            "- TripPassengerStats(TripID, passenger_count, avg_age, min_age, max_age, under18_count, age18_24_count, age20_25_count, over30_count)\n\n"
            "Notes:\n"
            "- TripDate is ISO ('YYYY-MM-DD HH:MM:SS'); use DATE(TripDate) and STRFTIME for filters.\n"
            "- Saturday = TripWeekday = 6; Weekend flag IsWeekend=1.\n"
            "- Downtown bars may appear in DropOffAddress with '6th' or 'Downtown'.\n"
            "- Moody Center via DropOffLabel='Moody Center' or DropOffAddress LIKE '%Moody%'.\n\n"
        )
        
        return schema_text
    
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
            
            # Format prompt
            prompt = self.sql_prompt_template.format(
                schema=schema_context,
                query=natural_query
            )
            
            # Call OpenAI API
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Generate clean, executable SQL queries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            # Extract SQL from response
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the SQL (remove markdown formatting if present)
            sql_query = re.sub(r'^```sql\s*', '', sql_query)
            sql_query = re.sub(r'\s*```$', '', sql_query)
            sql_query = sql_query.strip()
            
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
    
    def validate_and_enhance_query(self, sql_query: str, natural_query: str) -> Dict[str, Any]:
        """
        Validate and enhance SQL query with Fetii-specific improvements
        
        Args:
            sql_query: Generated SQL query
            natural_query: Original natural language query
            
        Returns:
            Dictionary with validation results and enhanced query
        """
        try:
            # Basic validation
            validation = self.validate_sql_query(sql_query)
            if not validation['valid']:
                return {
                    'valid': False,
                    'error': validation['error'],
                    'enhanced_query': sql_query
                }
            
            # Fetii-specific enhancements
            enhanced_query = sql_query
            
            # Add LIMIT if missing for large result sets
            if 'SELECT' in sql_query.upper() and 'LIMIT' not in sql_query.upper():
                if 'COUNT(' in sql_query.upper() or 'AVG(' in sql_query.upper() or 'SUM(' in sql_query.upper():
                    # Aggregation queries don't need LIMIT
                    pass
                else:
                    enhanced_query += ' LIMIT 100'
            
            # Ensure proper ordering for ranking queries
            if any(word in natural_query.lower() for word in ['top', 'most', 'highest', 'best']):
                if 'ORDER BY' not in sql_query.upper():
                    # Try to add ordering based on context
                    if 'COUNT(' in sql_query.upper():
                        enhanced_query += ' ORDER BY COUNT(*) DESC'
                    elif 'AVG(' in sql_query.upper():
                        enhanced_query += ' ORDER BY AVG(*) DESC'
            
            # Add proper aliases for better readability
            if 'SELECT COUNT(*)' in sql_query.upper():
                enhanced_query = sql_query.replace('COUNT(*)', 'COUNT(*) as count')
            
            return {
                'valid': True,
                'enhanced_query': enhanced_query,
                'original_query': sql_query,
                'improvements': ['Added LIMIT clause', 'Added proper ordering', 'Added column aliases']
            }
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return {
                'valid': False,
                'error': str(e),
                'enhanced_query': sql_query
            }
    
    def get_sample_queries(self) -> List[Dict[str, str]]:
        """Return sample queries for testing"""
        return [
            {
                'natural': 'How many groups went to Moody Center?',
                'sql': "SELECT COUNT(*) as moody_trips FROM TripData WHERE DropOffAddress LIKE '%Moody%'"
            },
            {
                'natural': 'Show me users older than 25',
                'sql': 'SELECT * FROM CustomerDemographics WHERE Age > 25'
            },
            {
                'natural': 'What is the average number of passengers per trip?',
                'sql': 'SELECT AVG(TotalPassengers) as avg_passengers FROM TripData'
            },
            {
                'natural': 'List all trips with more than 8 passengers',
                'sql': 'SELECT TripID, PickUpAddress, DropOffAddress, TotalPassengers FROM TripData WHERE TotalPassengers > 8'
            },
            {
                'natural': 'Show user demographics for trip 726765',
                'sql': '''SELECT cd.UserID, cd.Name, cd.Age, cd.Gender 
                         FROM CustomerDemographics cd 
                         JOIN CheckedInUsers ci ON cd.UserID = ci.UserID 
                         WHERE ci.TripID = 726765'''
            },
            {
                'natural': 'Count trips by user age groups',
                'sql': '''SELECT 
                            CASE 
                                WHEN cd.Age < 25 THEN 'Under 25'
                                WHEN cd.Age BETWEEN 25 AND 35 THEN '25-35'
                                WHEN cd.Age > 35 THEN 'Over 35'
                                ELSE 'Unknown'
                            END as age_group,
                            COUNT(DISTINCT t.TripID) as trip_count
                         FROM CustomerDemographics cd
                         JOIN CheckedInUsers ci ON cd.UserID = ci.UserID
                         JOIN TripData t ON ci.TripID = t.TripID
                         GROUP BY age_group'''
            },
            {
                'natural': 'How many groups went to Moody Center last month?',
                'sql': "SELECT COUNT(*) as moody_trips FROM TripData WHERE DropOffAddress LIKE '%Moody%' AND DATE(TripDate) >= DATE('now', '-1 month')"
            },
            {
                'natural': 'What are the top drop-off spots for 18-24 year-olds on Saturday nights?',
                'sql': '''SELECT t.DropOffAddress, COUNT(*) as trip_count
                         FROM TripData t
                         JOIN CheckedInUsers ci ON t.TripID = ci.TripID
                         JOIN CustomerDemographics cd ON ci.UserID = cd.UserID
                         WHERE cd.Age BETWEEN 18 AND 24 
                           AND strftime('%w', t.TripDate) = '6' 
                           AND strftime('%H', t.TripDate) >= '18'
                         GROUP BY t.DropOffAddress
                         ORDER BY trip_count DESC
                         LIMIT 10'''
            },
            {
                'natural': 'When do large groups (6+ riders) typically ride downtown?',
                'sql': '''SELECT strftime('%H', TripDate) as hour, COUNT(*) as trip_count
                         FROM TripData 
                         WHERE TotalPassengers >= 6 
                           AND (DropOffAddress LIKE '%downtown%' OR DropOffAddress LIKE '%6th%')
                         GROUP BY hour
                         ORDER BY trip_count DESC'''
            }
        ]
    
    def test_connection(self) -> Dict[str, Any]:
        """Test database connection and return basic stats"""
        try:
            with self.engine.connect() as conn:
                stats = {}
                
                # Get table counts
                for table in self.schema_info.keys():
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    stats[f"{table}_count"] = count_result.scalar()
                
                # Test a simple query
                test_query = "SELECT COUNT(*) as total FROM CustomerDemographics"
                test_result = conn.execute(text(test_query))
                
                return {
                    'success': True,
                    'connection_status': 'Connected',
                    'tables': list(self.schema_info.keys()),
                    'stats': stats,
                    'openai_available': bool(self.openai_api_key)
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'connection_status': 'Failed'
            }

def main():
    """Test the SQL Engine functionality"""
    # Initialize SQL Engine
    sql_engine = SQLEngine()
    
    print("Testing SQL Engine")
    print("=" * 50)
    
    # Test connection
    connection_test = sql_engine.test_connection()
    print(f"Connection Status: {connection_test}")
    print()
    
    if not connection_test['success']:
        print("Database connection failed. Cannot proceed with tests.")
        return
    
    # Test sample queries
    sample_queries = sql_engine.get_sample_queries()
    
    print("Testing Natural Language to SQL Translation:")
    print("-" * 50)
    
    for i, query_info in enumerate(sample_queries[:3], 1):  # Test first 3 queries
        natural_query = query_info['natural']
        expected_sql = query_info['sql']
        
        print(f"\n{i}. Natural Query: {natural_query}")
        
        # Test translation
        translation_result = sql_engine.translate_to_sql(natural_query)
        if translation_result['success']:
            print(f"   Generated SQL: {translation_result['sql_query']}")
        else:
            print(f"   Translation Error: {translation_result['error']}")
        
        # Test execution with expected SQL
        print(f"   Testing with expected SQL...")
        execution_result = sql_engine.execute_sql_query(expected_sql)
        if execution_result['success']:
            print(f"   Result: {execution_result['row_count']} rows returned")
            if hasattr(execution_result['data'], 'head'):
                print(f"   Sample data:\n{execution_result['data'].head()}")
        else:
            print(f"   Execution Error: {execution_result['error']}")
    
    # Test complete pipeline
    print(f"\n{'='*50}")
    print("Testing Complete Pipeline (NL → SQL → Results):")
    print("-" * 50)
    
    test_query = "How many users are in the database?"
    print(f"\nQuery: {test_query}")
    
    pipeline_result = sql_engine.process_natural_language_query(test_query)
    
    if pipeline_result['success']:
        print(f"Generated SQL: {pipeline_result['sql_query']}")
        print(f"Execution Time: {pipeline_result['execution_time']:.3f} seconds")
        print(f"Results: {pipeline_result['row_count']} rows")
        print(f"Data: {pipeline_result['data']}")
    else:
        print(f"Pipeline Error ({pipeline_result['stage']}): {pipeline_result['error']}")

if __name__ == "__main__":
    main()