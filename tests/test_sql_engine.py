import pytest
import pandas as pd
import os
import tempfile
import sqlite3
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.sql_engine import SQLEngine


class TestSQLEngine:
    """Test suite for the SQL Engine component"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary test database"""
        # Create temporary database file
        temp_db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db_path = temp_db_file.name
        temp_db_file.close()
        
        # Create test database with sample data
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE CustomerDemographics (
                UserID INTEGER PRIMARY KEY,
                Age INTEGER,
                Gender TEXT,
                Name TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE TripData (
                TripID INTEGER PRIMARY KEY,
                BookingUserID INTEGER,
                PickUpLatitude REAL,
                PickUpLongitude REAL,
                DropOffLatitude REAL,
                DropOffLongitude REAL,
                PickUpAddress TEXT,
                DropOffAddress TEXT,
                TripDate TEXT,
                TotalPassengers INTEGER,
                Duration REAL,
                Distance REAL,
                FOREIGN KEY (BookingUserID) REFERENCES CustomerDemographics(UserID)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE CheckedInUsers (
                CheckInID INTEGER PRIMARY KEY AUTOINCREMENT,
                UserID INTEGER,
                TripID INTEGER,
                CheckInTime TIMESTAMP,
                CheckInStatus TEXT DEFAULT 'active',
                FOREIGN KEY (UserID) REFERENCES CustomerDemographics(UserID),
                FOREIGN KEY (TripID) REFERENCES TripData(TripID)
            )
        ''')
        
        # Insert sample data
        sample_users = [
            (1, 25, 'Male', 'John Doe'),
            (2, 30, 'Female', 'Jane Smith'),
            (3, 35, 'Male', 'Bob Johnson'),
            (4, 22, 'Female', 'Alice Brown'),
            (5, 40, 'Male', 'Charlie Wilson')
        ]
        
        cursor.executemany(
            'INSERT INTO CustomerDemographics (UserID, Age, Gender, Name) VALUES (?, ?, ?, ?)',
            sample_users
        )
        
        sample_trips = [
            (101, 1, 31.5497, 74.3436, 31.5804, 74.3587, 'Lahore Station', 'Mall Road', '2024-01-01', 3, 25.5, 12.3),
            (102, 2, 31.5204, 74.3587, 31.4504, 74.2734, 'Gulberg', 'DHA', '2024-01-02', 5, 35.2, 18.7),
            (103, 3, 31.4697, 74.2728, 31.5497, 74.3436, 'Cantt', 'Liberty', '2024-01-03', 2, 20.1, 9.8),
            (104, 1, 31.5804, 74.3587, 31.5204, 74.3587, 'Mall Road', 'Anarkali', '2024-01-04', 8, 15.5, 5.2),
            (105, 4, 31.4504, 74.2734, 31.4697, 74.2728, 'DHA', 'Model Town', '2024-01-05', 12, 45.8, 22.1)
        ]
        
        cursor.executemany(
            '''INSERT INTO TripData (TripID, BookingUserID, PickUpLatitude, PickUpLongitude, 
               DropOffLatitude, DropOffLongitude, PickUpAddress, DropOffAddress, 
               TripDate, TotalPassengers, Duration, Distance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            sample_trips
        )
        
        sample_checkins = [
            (1, 101), (2, 102), (3, 102), (3, 103), 
            (1, 104), (4, 105), (5, 105)
        ]
        
        cursor.executemany(
            'INSERT INTO CheckedInUsers (UserID, TripID) VALUES (?, ?)',
            sample_checkins
        )
        
        conn.commit()
        conn.close()
        
        yield temp_db_path
        
        # Cleanup
        os.unlink(temp_db_path)
    
    @pytest.fixture
    def sql_engine(self, temp_db):
        """Create SQL engine instance with test database"""
        return SQLEngine(db_path=temp_db, openai_api_key="test-key")
    
    def test_database_connection(self, sql_engine):
        """Test database connection establishment"""
        assert sql_engine.engine is not None
        
        # Test connection
        connection_test = sql_engine.test_connection()
        assert connection_test['success'] == True
        assert 'CustomerDemographics' in connection_test['tables']
        assert 'TripData' in connection_test['tables']
        assert 'CheckedInUsers' in connection_test['tables']
    
    def test_schema_loading(self, sql_engine):
        """Test schema information loading"""
        assert len(sql_engine.schema_info) == 3
        assert 'CustomerDemographics' in sql_engine.schema_info
        assert 'TripData' in sql_engine.schema_info
        assert 'CheckedInUsers' in sql_engine.schema_info
        
        # Check column information
        demo_columns = [col['name'] for col in sql_engine.schema_info['CustomerDemographics']['columns']]
        assert 'UserID' in demo_columns
        assert 'Age' in demo_columns
        assert 'Gender' in demo_columns
        assert 'Name' in demo_columns
    
    def test_sql_validation_valid_queries(self, sql_engine):
        """Test SQL query validation with valid queries"""
        valid_queries = [
            "SELECT COUNT(*) FROM CustomerDemographics",
            "SELECT * FROM TripData WHERE TotalPassengers > 5",
            "SELECT AVG(Age) FROM CustomerDemographics",
            "SELECT t.TripID, c.Name FROM TripData t JOIN CustomerDemographics c ON t.BookingUserID = c.UserID"
        ]
        
        for query in valid_queries:
            validation = sql_engine.validate_sql_query(query)
            assert validation['valid'] == True, f"Query should be valid: {query}"
    
    def test_sql_validation_invalid_queries(self, sql_engine):
        """Test SQL query validation with dangerous queries"""
        dangerous_queries = [
            "DROP TABLE CustomerDemographics",
            "DELETE FROM TripData",
            "UPDATE CustomerDemographics SET Age = 0",
            "INSERT INTO TripData VALUES (999, 999, 0, 0, 0, 0, 'test', 'test', '2024', 1, 1, 1)",
            "ALTER TABLE CustomerDemographics ADD COLUMN test TEXT"
        ]
        
        for query in dangerous_queries:
            validation = sql_engine.validate_sql_query(query)
            assert validation['valid'] == False, f"Query should be invalid: {query}"
    
    def test_execute_sql_count_query(self, sql_engine):
        """Test executing count queries"""
        result = sql_engine.execute_sql_query("SELECT COUNT(*) as user_count FROM CustomerDemographics")
        
        assert result['success'] == True
        assert isinstance(result['data'], pd.DataFrame)
        assert len(result['data']) == 1
        assert result['data'].iloc[0]['user_count'] == 5
        assert result['row_count'] == 1
    
    def test_execute_sql_aggregation_query(self, sql_engine):
        """Test executing aggregation queries"""
        result = sql_engine.execute_sql_query("SELECT AVG(Age) as avg_age, COUNT(*) as total FROM CustomerDemographics")
        
        assert result['success'] == True
        assert isinstance(result['data'], pd.DataFrame)
        assert len(result['data']) == 1
        assert result['data'].iloc[0]['avg_age'] == 30.4  # (25+30+35+22+40)/5
        assert result['data'].iloc[0]['total'] == 5
    
    def test_execute_sql_filter_query(self, sql_engine):
        """Test executing filter queries"""
        result = sql_engine.execute_sql_query("SELECT * FROM TripData WHERE TotalPassengers > 5")
        
        assert result['success'] == True
        assert isinstance(result['data'], pd.DataFrame)
        assert len(result['data']) == 2  # Trips 104 (8 passengers) and 105 (12 passengers)
        assert all(result['data']['TotalPassengers'] > 5)
    
    def test_execute_sql_join_query(self, sql_engine):
        """Test executing JOIN queries"""
        query = """
        SELECT c.Name, c.Age, t.TripID, t.TotalPassengers 
        FROM CustomerDemographics c 
        JOIN TripData t ON c.UserID = t.BookingUserID 
        ORDER BY c.UserID
        """
        
        result = sql_engine.execute_sql_query(query)
        
        assert result['success'] == True
        assert isinstance(result['data'], pd.DataFrame)
        assert len(result['data']) == 5  # 5 trips total
        assert 'Name' in result['data'].columns
        assert 'Age' in result['data'].columns
        assert 'TripID' in result['data'].columns
        assert 'TotalPassengers' in result['data'].columns
    
    def test_execute_sql_complex_query(self, sql_engine):
        """Test executing complex analytical queries"""
        query = """
        SELECT 
            c.Gender,
            COUNT(t.TripID) as trip_count,
            AVG(t.Duration) as avg_duration,
            SUM(t.TotalPassengers) as total_passengers
        FROM CustomerDemographics c
        JOIN TripData t ON c.UserID = t.BookingUserID
        GROUP BY c.Gender
        ORDER BY trip_count DESC
        """
        
        result = sql_engine.execute_sql_query(query)
        
        assert result['success'] == True
        assert isinstance(result['data'], pd.DataFrame)
        assert len(result['data']) == 2  # Male and Female
        assert 'Gender' in result['data'].columns
        assert 'trip_count' in result['data'].columns
        assert 'avg_duration' in result['data'].columns
        assert 'total_passengers' in result['data'].columns
    
    def test_execute_invalid_sql(self, sql_engine):
        """Test executing invalid SQL"""
        result = sql_engine.execute_sql_query("SELECT * FROM NonExistentTable")
        
        assert result['success'] == False
        assert 'error' in result
        assert result['data'] is None
    
    @patch('openai.ChatCompletion.create')
    def test_translate_to_sql_success(self, mock_openai, sql_engine):
        """Test successful natural language to SQL translation"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "SELECT COUNT(*) FROM CustomerDemographics"
        mock_openai.return_value = mock_response
        
        result = sql_engine.translate_to_sql("How many users are in the database?")
        
        assert result['success'] == True
        assert result['sql_query'] == "SELECT COUNT(*) FROM CustomerDemographics"
        assert result['method'] == 'openai_translation'
        mock_openai.assert_called_once()
    
    def test_translate_to_sql_no_api_key(self):
        """Test translation without API key"""
        sql_engine_no_key = SQLEngine(openai_api_key=None)
        result = sql_engine_no_key.translate_to_sql("How many users are there?")
        
        assert result['success'] == False
        assert 'API key not available' in result['error']
    
    @patch('openai.ChatCompletion.create')
    def test_process_natural_language_query_success(self, mock_openai, sql_engine):
        """Test complete natural language query processing pipeline"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "SELECT COUNT(*) as total FROM CustomerDemographics"
        mock_openai.return_value = mock_response
        
        result = sql_engine.process_natural_language_query("How many users are in the database?")
        
        assert result['success'] == True
        assert result['natural_query'] == "How many users are in the database?"
        assert result['sql_query'] == "SELECT COUNT(*) as total FROM CustomerDemographics"
        assert isinstance(result['data'], pd.DataFrame)
        assert result['data'].iloc[0]['total'] == 5
    
    @patch('openai.ChatCompletion.create')
    def test_process_natural_language_query_sql_error(self, mock_openai, sql_engine):
        """Test pipeline with SQL execution error"""
        # Mock OpenAI response with invalid SQL
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "SELECT * FROM InvalidTable"
        mock_openai.return_value = mock_response
        
        result = sql_engine.process_natural_language_query("Show me data from invalid table")
        
        assert result['success'] == False
        assert result['stage'] == 'execution_failed'
        assert 'error' in result
    
    def test_sample_queries(self, sql_engine):
        """Test provided sample queries"""
        sample_queries = sql_engine.get_sample_queries()
        
        assert len(sample_queries) > 0
        
        # Test a few sample queries
        for sample in sample_queries[:3]:  # Test first 3
            result = sql_engine.execute_sql_query(sample['sql'])
            assert result['success'] == True, f"Sample query failed: {sample['sql']}"
    
    def test_get_schema_context(self, sql_engine):
        """Test schema context generation"""
        schema_context = sql_engine.get_schema_context()
        
        assert isinstance(schema_context, str)
        assert 'CustomerDemographics' in schema_context
        assert 'TripData' in schema_context
        assert 'CheckedInUsers' in schema_context
        assert 'UserID' in schema_context
        assert 'TripID' in schema_context


# Additional integration tests
class TestSQLEngineIntegration:
    """Integration tests for SQL Engine with real-world scenarios"""
    
    @pytest.fixture
    def sql_engine_with_data(self, temp_db):
        """SQL engine with comprehensive test data"""
        return SQLEngine(db_path=temp_db, openai_api_key="test-key")
    
    def test_user_analytics_queries(self, sql_engine_with_data):
        """Test user analytics queries"""
        engine = sql_engine_with_data
        
        # Age distribution
        result = engine.execute_sql_query("""
            SELECT 
                CASE 
                    WHEN Age < 25 THEN 'Under 25'
                    WHEN Age BETWEEN 25 AND 35 THEN '25-35'
                    ELSE 'Over 35'
                END as age_group,
                COUNT(*) as count
            FROM CustomerDemographics
            GROUP BY age_group
        """)
        
        assert result['success'] == True
        assert len(result['data']) >= 2  # Should have multiple age groups
    
    def test_trip_analytics_queries(self, sql_engine_with_data):
        """Test trip analytics queries"""
        engine = sql_engine_with_data
        
        # Trip statistics
        result = engine.execute_sql_query("""
            SELECT 
                COUNT(*) as total_trips,
                AVG(Duration) as avg_duration,
                AVG(Distance) as avg_distance,
                AVG(TotalPassengers) as avg_passengers,
                MAX(TotalPassengers) as max_passengers,
                MIN(TotalPassengers) as min_passengers
            FROM TripData
        """)
        
        assert result['success'] == True
        assert len(result['data']) == 1
        data = result['data'].iloc[0]
        assert data['total_trips'] == 5
        assert data['max_passengers'] == 12
        assert data['min_passengers'] == 2
    
    def test_user_trip_relationship_queries(self, sql_engine_with_data):
        """Test queries involving user-trip relationships"""
        engine = sql_engine_with_data
        
        # User trip counts
        result = engine.execute_sql_query("""
            SELECT 
                c.Name,
                COUNT(ci.TripID) as checkin_count,
                COUNT(DISTINCT t.TripID) as unique_trips
            FROM CustomerDemographics c
            LEFT JOIN CheckedInUsers ci ON c.UserID = ci.UserID
            LEFT JOIN TripData t ON c.UserID = t.BookingUserID
            GROUP BY c.UserID, c.Name
            ORDER BY checkin_count DESC
        """)
        
        assert result['success'] == True
        assert len(result['data']) == 5  # All 5 users
        assert 'Name' in result['data'].columns
        assert 'checkin_count' in result['data'].columns


def test_sql_engine_error_handling():
    """Test SQL engine error handling with invalid database path"""
    with pytest.raises(Exception):
        SQLEngine(db_path="nonexistent_database.db")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])