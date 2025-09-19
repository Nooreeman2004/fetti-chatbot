import pytest
import pandas as pd
import tempfile
import os
import pickle
import json
import faiss
import numpy as np
import sqlite3
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.hybrid_controller import HybridController
from chatbot.query_classifier import QueryType


class TestHybridController:
    """Test suite for the Hybrid Controller component"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary test database"""
        temp_db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db_path = temp_db_file.name
        temp_db_file.close()
        
        # Create test database
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
        os.unlink(temp_db_path)
    
    @pytest.fixture
    def temp_embeddings_dir(self):
        """Create temporary embeddings directory"""
        temp_dir = tempfile.mkdtemp()
        
        # Create mock FAISS index
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        num_vectors = 10
        sample_embeddings = np.random.random((num_vectors, dimension)).astype(np.float32)
        faiss.normalize_L2(sample_embeddings)
        index.add(sample_embeddings)
        
        with open(os.path.join(temp_dir, 'faiss_index.pkl'), 'wb') as f:
            pickle.dump(index, f)
        
        # Create sample metadata
        metadata = []
        text_representations = []
        for i in range(num_vectors):
            if i % 2 == 0:
                metadata.append({
                    'type': 'demographics',
                    'user_id': 1 + i//2,
                    'source_table': 'CustomerDemographics'
                })
                text_representations.append(f'Customer profile: User ID: {1 + i//2}, Age: {25 + i}, Gender: Male')
            else:
                metadata.append({
                    'type': 'trip',
                    'trip_id': 101 + i//2,
                    'source_table': 'TripData'
                })
                text_representations.append(f'Transportation trip: Trip ID: {101 + i//2}, Passengers: {3 + i}')
        
        with open(os.path.join(temp_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        with open(os.path.join(temp_dir, 'text_representations.pkl'), 'wb') as f:
            pickle.dump(text_representations, f)
        
        # Create configuration
        config = {
            'model_name': 'all-MiniLM-L6-v2',
            'embedding_dimension': dimension,
            'total_vectors': num_vectors,
            'creation_timestamp': '2024-01-01T00:00:00'
        }
        
        with open(os.path.join(temp_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        yield temp_dir
        
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def hybrid_controller(self, temp_db, temp_embeddings_dir):
        """Create hybrid controller with test data"""
        return HybridController(
            db_path=temp_db,
            embeddings_dir=temp_embeddings_dir,
            openai_api_key="test-key"
        )
    
    def test_initialization(self, hybrid_controller):
        """Test hybrid controller initialization"""
        assert hybrid_controller.db_path is not None
        assert hybrid_controller.embeddings_dir is not None
        assert hybrid_controller.openai_api_key == "test-key"
        assert hybrid_controller.default_top_k == 5
        assert hybrid_controller.confidence_threshold == 0.5
    
    def test_get_system_status(self, hybrid_controller):
        """Test system status reporting"""
        status = hybrid_controller.get_system_status()
        
        assert 'timestamp' in status
        assert 'components' in status
        assert 'overall_status' in status
        
        # Check for expected components
        expected_components = ['query_classifier', 'sql_engine', 'rag_engine']
        for component in expected_components:
            assert component in status['components']
            assert 'status' in status['components'][component]
    
    @patch('chatbot.hybrid_controller.QueryClassifier')
    def test_classify_query_success(self, mock_classifier_class, hybrid_controller):
        """Test successful query classification"""
        # Mock the classifier instance
        mock_classifier = Mock()
        mock_classifier.classify_query.return_value = {
            'classification': QueryType.SQL,
            'confidence': 0.8,
            'method': 'rule_based'
        }
        mock_classifier_class.return_value = mock_classifier
        
        # Replace the controller's classifier
        hybrid_controller.query_classifier = mock_classifier
        
        result = hybrid_controller.classify_query("How many users are there?")
        
        assert result['classification'] == QueryType.SQL
        assert result['confidence'] == 0.8
        assert result['method'] == 'rule_based'
    
    def test_classify_query_no_classifier(self, temp_db, temp_embeddings_dir):
        """Test query classification when classifier is not available"""
        controller = HybridController(
            db_path=temp_db,
            embeddings_dir=temp_embeddings_dir,
            openai_api_key=None
        )
        controller.query_classifier = None
        
        result = controller.classify_query("test query")
        
        assert result['classification'] is None
        assert result['confidence'] == 0.0
        assert result['method'] == 'classifier_unavailable'
        assert 'error' in result
    
    @patch('chatbot.hybrid_controller.SQLEngine')
    def test_process_sql_query_success(self, mock_sql_engine_class, hybrid_controller):
        """Test successful SQL query processing"""
        # Mock SQL engine
        mock_sql_engine = Mock()
        mock_sql_engine.process_natural_language_query.return_value = {
            'success': True,
            'sql_query': 'SELECT COUNT(*) FROM CustomerDemographics',
            'data': pd.DataFrame({'count': [5]}),
            'row_count': 1,
            'execution_time': 0.05
        }
        mock_sql_engine_class.return_value = mock_sql_engine
        
        # Replace controller's SQL engine
        hybrid_controller.sql_engine = mock_sql_engine
        
        result = hybrid_controller.process_sql_query("How many users are there?")
        
        assert result['success'] == True
        assert result['query_type'] == 'SQL'
        assert 'sql_query' in result
        assert 'data' in result
    
    def test_process_sql_query_no_engine(self, temp_db, temp_embeddings_dir):
        """Test SQL processing when engine is not available"""
        controller = HybridController(
            db_path=temp_db,
            embeddings_dir=temp_embeddings_dir,
            openai_api_key=None
        )
        controller.sql_engine = None
        
        result = controller.process_sql_query("test query")
        
        assert result['success'] == False
        assert 'SQL Engine not initialized' in result['error']
        assert result['query_type'] == 'SQL'
    
    @patch('chatbot.hybrid_controller.RAGEngine')
    def test_process_rag_query_success(self, mock_rag_engine_class, hybrid_controller):
        """Test successful RAG query processing"""
        # Mock RAG engine
        mock_rag_engine = Mock()
        mock_rag_engine.retrieve_context.return_value = {
            'query': 'test query',
            'returned_results': 3,
            'total_results': 5,
            'context_snippets': [
                {'rank': 1, 'relevance_score': 0.85, 'content': 'Test content'},
                {'rank': 2, 'relevance_score': 0.72, 'content': 'More content'}
            ],
            'search_metadata': {'model': 'test-model'}
        }
        mock_rag_engine_class.return_value = mock_rag_engine
        
        # Replace controller's RAG engine
        hybrid_controller.rag_engine = mock_rag_engine
        
        result = hybrid_controller.process_rag_query("tell me about user 123")
        
        assert result['success'] == True
        assert result['query_type'] == 'RAG'
        assert 'context_snippets' in result
        assert result['returned_results'] == 3
    
    def test_process_rag_query_no_engine(self, temp_db, temp_embeddings_dir):
        """Test RAG processing when engine is not available"""
        controller = HybridController(
            db_path=temp_db,
            embeddings_dir=temp_embeddings_dir,
            openai_api_key=None
        )
        controller.rag_engine = None
        
        result = controller.process_rag_query("test query")
        
        assert result['success'] == False
        assert 'RAG Engine not initialized' in result['error']
        assert result['query_type'] == 'RAG'
    
    @patch('chatbot.hybrid_controller.SQLEngine')
    @patch('chatbot.hybrid_controller.RAGEngine')
    def test_process_hybrid_query_success(self, mock_rag_engine_class, mock_sql_engine_class, hybrid_controller):
        """Test successful hybrid query processing"""
        # Mock SQL engine
        mock_sql_engine = Mock()
        mock_sql_engine.process_natural_language_query.return_value = {
            'success': True,
            'sql_query': 'SELECT * FROM CustomerDemographics WHERE Age > 25',
            'data': pd.DataFrame({
                'UserID': [2, 3, 5],
                'Age': [30, 35, 40],
                'Name': ['Jane Smith', 'Bob Johnson', 'Charlie Wilson']
            }),
            'row_count': 3
        }
        
        # Mock RAG engine
        mock_rag_engine = Mock()
        mock_rag_engine.retrieve_context.return_value = {
            'query': 'test query',
            'returned_results': 2,
            'total_results': 5,
            'context_snippets': [
                {'rank': 1, 'relevance_score': 0.85, 'content': 'User profile content'},
                {'rank': 2, 'relevance_score': 0.72, 'content': 'Trip information'}
            ]
        }
        
        # Replace engines
        hybrid_controller.sql_engine = mock_sql_engine
        hybrid_controller.rag_engine = mock_rag_engine
        
        result = hybrid_controller.process_hybrid_query("Show me users over 25 and their details")
        
        assert result['success'] == True
        assert result['query_type'] == 'HYBRID'
        assert 'sql_result' in result
        assert 'rag_result' in result
        assert 'combined_insights' in result
        assert result['sql_result']['success'] == True
        assert result['rag_result']['success'] == True
    
    def test_extract_entities_for_rag_enhancement(self, hybrid_controller):
        """Test entity extraction from SQL results"""
        # Create test DataFrame
        test_data = pd.DataFrame({
            'UserID': [1, 2, 3],
            'TripID': [101, 102, 103],
            'PickUpAddress': ['Location A', 'Location B', 'Location C']
        })
        
        enhanced_queries = hybrid_controller._extract_entities_for_rag_enhancement(test_data)
        
        assert isinstance(enhanced_queries, list)
        
        # Should generate queries for UserIDs, TripIDs, and addresses
        user_queries = [q for q in enhanced_queries if 'user' in q.lower()]
        trip_queries = [q for q in enhanced_queries if 'trip' in q.lower()]
        
        assert len(user_queries) > 0
        assert len(trip_queries) > 0
    
    def test_generate_combined_insights(self, hybrid_controller):
        """Test combined insights generation"""
        sql_result = {
            'success': True,
            'data': pd.DataFrame({
                'UserID': [1, 2],
                'Age': [25, 30],
                'Name': ['John', 'Jane']
            }),
            'execution_time': 0.05
        }
        
        rag_result = {
            'success': True,
            'context_snippets': [
                {'relevance_score': 0.85, 'type': 'demographics'},
                {'relevance_score': 0.72, 'type': 'trip'}
            ]
        }
        
        insights = hybrid_controller._generate_combined_insights(sql_result, rag_result)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        # Check insight structure
        for insight in insights:
            assert 'type' in insight
            assert 'title' in insight
            assert 'content' in insight
    
    @patch('chatbot.hybrid_controller.QueryClassifier')
    @patch('chatbot.hybrid_controller.SQLEngine')
    @patch('chatbot.hybrid_controller.RAGEngine')
    def test_process_query_sql_routing(self, mock_rag_class, mock_sql_class, mock_classifier_class, hybrid_controller):
        """Test query routing for SQL queries"""
        # Mock classifier to return SQL
        mock_classifier = Mock()
        mock_classifier.classify_query.return_value = {
            'classification': QueryType.SQL,
            'confidence': 0.8,
            'method': 'rule_based'
        }
        
        # Mock SQL engine
        mock_sql_engine = Mock()
        mock_sql_engine.process_natural_language_query.return_value = {
            'success': True,
            'sql_query': 'SELECT COUNT(*) FROM CustomerDemographics',
            'data': pd.DataFrame({'count': [5]}),
            'row_count': 1
        }
        
        # Replace components
        hybrid_controller.query_classifier = mock_classifier
        hybrid_controller.sql_engine = mock_sql_engine
        
        result = hybrid_controller.process_query("How many users are there?")
        
        assert result['success'] == True
        assert result['processing_type'] == 'SQL'
        assert 'classification' in result
        assert result['classification']['classification'] == QueryType.SQL
    
    @patch('chatbot.hybrid_controller.QueryClassifier')
    @patch('chatbot.hybrid_controller.RAGEngine')
    def test_process_query_rag_routing(self, mock_rag_class, mock_classifier_class, hybrid_controller):
        """Test query routing for RAG queries"""
        # Mock classifier to return RAG
        mock_classifier = Mock()
        mock_classifier.classify_query.return_value = {
            'classification': QueryType.RAG,
            'confidence': 0.9,
            'method': 'rule_based'
        }
        
        # Mock RAG engine
        mock_rag_engine = Mock()
        mock_rag_engine.retrieve_context.return_value = {
            'query': 'tell me about user 123',
            'returned_results': 2,
            'total_results': 5,
            'context_snippets': [
                {'rank': 1, 'content': 'User profile information'}
            ]
        }
        
        # Replace components
        hybrid_controller.query_classifier = mock_classifier
        hybrid_controller.rag_engine = mock_rag_engine
        
        result = hybrid_controller.process_query("tell me about user 123")
        
        assert result['success'] == True
        assert result['processing_type'] == 'RAG'
        assert result['classification']['classification'] == QueryType.RAG
    
    def test_process_query_forced_type(self, hybrid_controller):
        """Test query processing with forced type"""
        # Mock engines
        mock_sql_engine = Mock()
        mock_sql_engine.process_natural_language_query.return_value = {
            'success': True,
            'data': pd.DataFrame({'result': ['test']})
        }
        hybrid_controller.sql_engine = mock_sql_engine
        
        result = hybrid_controller.process_query("test query", force_type="SQL")
        
        assert result['processing_type'] == 'SQL'
        assert result['classification']['method'] == 'forced'
        assert result['classification']['confidence'] == 1.0
    
    def test_process_query_classification_failure(self, temp_db, temp_embeddings_dir):
        """Test query processing when classification fails"""
        controller = HybridController(
            db_path=temp_db,
            embeddings_dir=temp_embeddings_dir,
            openai_api_key=None
        )
        controller.query_classifier = None
        
        result = controller.process_query("test query")
        
        assert result['success'] == False
        assert 'Query classification failed' in result['error']
        assert result['classification']['classification'] is None
    
    def test_process_query_timing(self, hybrid_controller):
        """Test that query processing includes timing information"""
        # Mock a simple successful SQL query
        mock_sql_engine = Mock()
        mock_sql_engine.process_natural_language_query.return_value = {
            'success': True,
            'data': pd.DataFrame({'result': ['test']})
        }
        hybrid_controller.sql_engine = mock_sql_engine
        
        result = hybrid_controller.process_query("test", force_type="SQL")
        
        assert 'processing_time' in result
        assert 'timestamp' in result
        assert isinstance(result['processing_time'], float)
        assert result['processing_time'] >= 0


class TestHybridControllerIntegration:
    """Integration tests for Hybrid Controller with realistic scenarios"""
    
    @pytest.fixture
    def integration_controller(self, temp_db, temp_embeddings_dir):
        """Create controller for integration testing"""
        return HybridController(
            db_path=temp_db,
            embeddings_dir=temp_embeddings_dir,
            openai_api_key="test-key",
            default_top_k=3,
            confidence_threshold=0.6
        )
    
    def test_end_to_end_sql_workflow(self, integration_controller):
        """Test complete SQL workflow without mocking"""
        # This would test the actual SQL engine if it was properly initialized
        # For now, we'll test the flow structure
        query = "How many users are in the database?"
        
        # The actual execution might fail due to OpenAI API key, but we can test the structure
        result = integration_controller.process_query(query, force_type="SQL")
        
        assert 'success' in result
        assert 'processing_type' in result
        assert 'processing_time' in result
        assert result['processing_type'] == 'SQL'
    
    def test_end_to_end_rag_workflow(self, integration_controller):
        """Test complete RAG workflow"""
        query = "tell me about user profile"
        
        # Force RAG type to test RAG engine integration
        result = integration_controller.process_query(query, force_type="RAG")
        
        assert 'success' in result
        assert 'processing_type' in result
        assert result['processing_type'] == 'RAG'
    
    def test_end_to_end_hybrid_workflow(self, integration_controller):
        """Test complete hybrid workflow"""
        query = "show me user statistics and tell me about specific users"
        
        # Force hybrid type
        result = integration_controller.process_query(query, force_type="HYBRID")
        
        assert 'success' in result
        assert 'processing_type' in result
        assert result['processing_type'] == 'HYBRID'
    
    def test_system_resilience(self, temp_db, temp_embeddings_dir):
        """Test system behavior when components fail to initialize"""
        # Test with missing OpenAI key
        controller = HybridController(
            db_path=temp_db,
            embeddings_dir=temp_embeddings_dir,
            openai_api_key=None
        )
        
        status = controller.get_system_status()
        assert status['overall_status'] == 'degraded'
        
        # Should still be able to process some queries
        result = controller.process_query("test query", force_type="RAG")
        assert 'success' in result
    
    def test_multiple_query_processing(self, integration_controller):
        """Test processing multiple queries in sequence"""
        queries = [
            ("How many users?", "SQL"),
            ("Tell me about users", "RAG"),
            ("Analyze user patterns", "HYBRID")
        ]
        
        results = []
        for query, query_type in queries:
            result = integration_controller.process_query(query, force_type=query_type)
            results.append(result)
            
            # Each result should have consistent structure
            assert 'success' in result
            assert 'processing_type' in result
            assert 'processing_time' in result
            assert result['processing_type'] == query_type
        
        # All results should be processed
        assert len(results) == 3
    
    def test_parameter_variations(self, integration_controller):
        """Test processing with different parameter combinations"""
        query = "test query"
        
        # Test with different top_k values
        result1 = integration_controller.process_query(query, force_type="RAG", top_k=3)
        result2 = integration_controller.process_query(query, force_type="RAG", top_k=5)
        
        assert result1['processing_type'] == 'RAG'
        assert result2['processing_type'] == 'RAG'


class TestHybridControllerErrorHandling:
    """Test error handling scenarios"""
    
    def test_invalid_database_path(self):
        """Test behavior with invalid database path"""
        with pytest.raises(Exception):
            HybridController(
                db_path="nonexistent/path/database.db",
                embeddings_dir="valid_dir",
                openai_api_key="test-key"
            )
    
    def test_invalid_embeddings_directory(self, temp_db):
        """Test behavior with invalid embeddings directory"""
        with pytest.raises(Exception):
            HybridController(
                db_path=temp_db,
                embeddings_dir="nonexistent/embeddings/dir",
                openai_api_key="test-key"
            )
    
    def test_empty_query_handling(self, temp_db, temp_embeddings_dir):
        """Test handling of empty queries"""
        controller = HybridController(
            db_path=temp_db,
            embeddings_dir=temp_embeddings_dir,
            openai_api_key="test-key"
        )
        
        result = controller.process_query("")
        
        assert result['success'] == False
        assert 'error' in result
    
    def test_malformed_query_handling(self, temp_db, temp_embeddings_dir):
        """Test handling of malformed queries"""
        controller = HybridController(
            db_path=temp_db,
            embeddings_dir=temp_embeddings_dir,
            openai_api_key="test-key"
        )
        
        # Test with very long query
        long_query = "test " * 1000
        result = controller.process_query(long_query, force_type="SQL")
        
        # Should not crash, may succeed or fail gracefully
        assert 'success' in result
        assert 'processing_time' in result


def test_main_function():
    """Test the main function exists and can be called"""
    # Import and test the main function
    try:
        from chatbot.hybrid_controller import main
        # Don't actually run it as it might create real database connections
        assert callable(main)
    except ImportError:
        pytest.skip("Main function not available for testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])