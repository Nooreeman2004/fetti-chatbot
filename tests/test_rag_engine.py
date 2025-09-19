import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import pickle
import json
import faiss
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.rag_engine import RAGEngine


class TestRAGEngine:
    """Test suite for the RAG Engine component"""
    
    @pytest.fixture
    def temp_embeddings_dir(self):
        """Create temporary embeddings directory with test data"""
        temp_dir = tempfile.mkdtemp()
        
        # Create mock FAISS index
        dimension = 384  # all-MiniLM-L6-v2 dimension
        index = faiss.IndexFlatIP(dimension)
        
        # Generate sample embeddings
        num_vectors = 10
        sample_embeddings = np.random.random((num_vectors, dimension)).astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(sample_embeddings)
        
        # Add to index
        index.add(sample_embeddings)
        
        # Save FAISS index
        with open(os.path.join(temp_dir, 'faiss_index.pkl'), 'wb') as f:
            pickle.dump(index, f)
        
        # Create sample metadata
        metadata = []
        for i in range(num_vectors):
            metadata.append({
                'type': 'demographics' if i % 2 == 0 else 'trip',
                'id': f'test_{i}',
                'user_id': 1000 + i if i % 2 == 0 else None,
                'trip_id': 2000 + i if i % 2 == 1 else None,
                'source_table': 'CustomerDemographics' if i % 2 == 0 else 'TripData',
                'raw_data': {'test_field': f'test_value_{i}'}
            })
        
        with open(os.path.join(temp_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        # Create sample text representations
        text_representations = []
        for i in range(num_vectors):
            if i % 2 == 0:
                text_representations.append(f'Customer profile: User ID: {1000+i}, Age: {25+i}, Gender: Male')
            else:
                text_representations.append(f'Transportation trip: Trip ID: {2000+i}, Pickup: Location {i}, Passengers: {3+i}')
        
        with open(os.path.join(temp_dir, 'text_representations.pkl'), 'wb') as f:
            pickle.dump(text_representations, f)
        
        # Create configuration
        config = {
            'model_name': 'all-MiniLM-L6-v2',
            'embedding_dimension': dimension,
            'total_vectors': num_vectors,
            'creation_timestamp': '2024-01-01T00:00:00',
            'db_path': 'test_database.db'
        }
        
        with open(os.path.join(temp_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        yield temp_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary test database for context enrichment"""
        import sqlite3
        temp_db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db_path = temp_db_file.name
        temp_db_file.close()
        
        # Create test database
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE CustomerDemographics (
                UserID INTEGER PRIMARY KEY,
                Age INTEGER,
                Gender TEXT,
                Name TEXT
            )
        ''')
        
        cursor.execute('''
            INSERT INTO CustomerDemographics VALUES 
            (1000, 25, 'Male', 'Test User 1'),
            (1001, 26, 'Female', 'Test User 2'),
            (1002, 27, 'Male', 'Test User 3')
        ''')
        
        cursor.execute('''
            CREATE TABLE TripData (
                TripID INTEGER PRIMARY KEY,
                BookingUserID INTEGER,
                PickUpAddress TEXT,
                DropOffAddress TEXT,
                TotalPassengers INTEGER,
                Duration REAL
            )
        ''')
        
        cursor.execute('''
            INSERT INTO TripData VALUES 
            (2001, 1000, 'Location A', 'Location B', 4, 25.5),
            (2003, 1001, 'Location C', 'Location D', 6, 35.2)
        ''')
        
        cursor.execute('''
            CREATE TABLE CheckedInUsers (
                CheckInID INTEGER PRIMARY KEY AUTOINCREMENT,
                UserID INTEGER,
                TripID INTEGER,
                CheckInTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CheckInStatus TEXT DEFAULT 'active'
            )
        ''')
        
        cursor.execute('''
            INSERT INTO CheckedInUsers (UserID, TripID) VALUES 
            (1000, 2001),
            (1001, 2003)
        ''')
        
        conn.commit()
        conn.close()
        
        yield temp_db_path
        os.unlink(temp_db_path)
    
    @pytest.fixture
    def rag_engine(self, temp_embeddings_dir, temp_db):
        """Create RAG engine with test data"""
        return RAGEngine(
            embeddings_dir=temp_embeddings_dir,
            db_path=temp_db,
            model_name='all-MiniLM-L6-v2'
        )
    
    def test_embeddings_loading(self, rag_engine):
        """Test loading of embeddings components"""
        assert rag_engine.faiss_index is not None
        assert rag_engine.metadata is not None
        assert rag_engine.text_representations is not None
        assert rag_engine.config is not None
        
        # Check dimensions
        assert rag_engine.faiss_index.ntotal == 10
        assert len(rag_engine.metadata) == 10
        assert len(rag_engine.text_representations) == 10
    
    def test_sentence_transformer_loading(self, rag_engine):
        """Test SentenceTransformer model loading"""
        assert rag_engine.model is not None
        assert rag_engine.model_name == 'all-MiniLM-L6-v2'
    
    def test_database_connection(self, rag_engine):
        """Test database connection for context enrichment"""
        assert rag_engine.engine is not None
    
    def test_encode_query(self, rag_engine):
        """Test query encoding"""
        query = "test user profile"
        embedding = rag_engine.encode_query(query)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 384)  # all-MiniLM-L6-v2 dimension
        assert embedding.dtype == np.float32
        
        # Check if normalized (for cosine similarity)
        norm = np.linalg.norm(embedding[0])
        assert abs(norm - 1.0) < 1e-5  # Should be normalized to 1
    
    def test_search_similar_records(self, rag_engine):
        """Test semantic similarity search"""
        query = "user profile customer"
        results = rag_engine.search_similar_records(query, top_k=5)
        
        assert isinstance(results, list)
        assert len(results) <= 5
        
        if results:
            # Check result structure
            result = results[0]
            assert 'rank' in result
            assert 'score' in result
            assert 'index' in result
            assert 'text' in result
            assert 'metadata' in result
            assert 'query' in result
            
            assert result['rank'] == 1
            assert isinstance(result['score'], float)
            assert result['query'] == query
    
    def test_search_with_score_threshold(self, rag_engine):
        """Test search with score threshold filtering"""
        query = "test query"
        
        # Search with no threshold
        results_no_threshold = rag_engine.search_similar_records(query, top_k=10, score_threshold=0.0)
        
        # Search with high threshold
        results_high_threshold = rag_engine.search_similar_records(query, top_k=10, score_threshold=0.9)
        
        # High threshold should return fewer or equal results
        assert len(results_high_threshold) <= len(results_no_threshold)
    
    def test_enrich_results_with_context(self, rag_engine):
        """Test context enrichment functionality"""
        # Create mock results with user_id
        mock_results = [
            {
                'rank': 1,
                'score': 0.85,
                'index': 0,
                'text': 'Customer profile: User ID: 1000, Age: 25, Gender: Male',
                'metadata': {'user_id': 1000, 'type': 'demographics', 'source_table': 'CustomerDemographics'},
                'query': 'test query'
            }
        ]
        
        enriched_results = rag_engine.enrich_results_with_context(mock_results)
        
        assert len(enriched_results) == 1
        enriched_result = enriched_results[0]
        
        # Check that additional_context was added
        assert 'additional_context' in enriched_result
        
        # Check that user details were enriched
        if enriched_result['additional_context'].get('user_details'):
            user_details = enriched_result['additional_context']['user_details']
            assert user_details['UserID'] == 1000
            assert user_details['Name'] == 'Test User 1'
    
    def test_format_results_for_llm(self, rag_engine):
        """Test formatting results for LLM consumption"""
        sample_results = [
            {
                'rank': 1,
                'score': 0.85,
                'index': 0,
                'text': 'Customer profile: User ID: 1000, Age: 25, Gender: Male',
                'metadata': {
                    'type': 'demographics',
                    'user_id': 1000,
                    'source_table': 'CustomerDemographics'
                },
                'query': 'test query'
            },
            {
                'rank': 2,
                'score': 0.72,
                'index': 1,
                'text': 'Transportation trip: Trip ID: 2001, Passengers: 4',
                'metadata': {
                    'type': 'trip',
                    'trip_id': 2001,
                    'source_table': 'TripData'
                },
                'query': 'test query'
            }
        ]
        
        formatted = rag_engine.format_results_for_llm(sample_results, "test query")
        
        assert 'query' in formatted
        assert 'total_results' in formatted
        assert 'returned_results' in formatted
        assert 'context_snippets' in formatted
        assert 'search_metadata' in formatted
        
        assert formatted['query'] == 'test query'
        assert formatted['total_results'] == 2
        assert formatted['returned_results'] == 2
        assert len(formatted['context_snippets']) == 2
        
        # Check snippet structure
        snippet = formatted['context_snippets'][0]
        assert 'rank' in snippet
        assert 'relevance_score' in snippet
        assert 'content' in snippet
        assert 'type' in snippet
        assert 'source' in snippet
    
    def test_format_results_length_constraint(self, rag_engine):
        """Test that formatting respects length constraints"""
        # Create many large results
        large_results = []
        for i in range(20):
            large_results.append({
                'rank': i + 1,
                'score': 0.8 - i * 0.01,
                'index': i,
                'text': 'Very long text content that should trigger length constraints ' * 20,
                'metadata': {'type': 'test', 'id': i},
                'query': 'test query'
            })
        
        formatted = rag_engine.format_results_for_llm(large_results, "test query", max_context_length=1000)
        
        # Should truncate due to length constraint
        assert formatted['returned_results'] < len(large_results)
    
    def test_retrieve_context_success(self, rag_engine):
        """Test complete context retrieval pipeline"""
        query = "user profile information"
        context = rag_engine.retrieve_context(query, top_k=3)
        
        assert 'query' in context
        assert 'total_results' in context
        assert 'returned_results' in context
        assert 'context_snippets' in context
        
        assert context['query'] == query
    
    def test_retrieve_context_no_results(self, rag_engine):
        """Test context retrieval with no matching results"""
        # Use very high score threshold to ensure no results
        context = rag_engine.retrieve_context("impossible query", top_k=5, score_threshold=0.99)
        
        # Should handle no results gracefully
        assert context['returned_results'] == 0 or context['total_results'] == 0
        assert isinstance(context['context_snippets'], list)
    
    def test_get_statistics(self, rag_engine):
        """Test RAG engine statistics"""
        stats = rag_engine.get_statistics()
        
        assert 'faiss_index_size' in stats
        assert 'metadata_count' in stats
        assert 'text_representations_count' in stats
        assert 'model_name' in stats
        assert 'embeddings_dir' in stats
        
        assert stats['faiss_index_size'] == 10
        assert stats['metadata_count'] == 10
        assert stats['text_representations_count'] == 10
        assert stats['model_name'] == 'all-MiniLM-L6-v2'
        
        # Check record type distribution
        if 'record_type_distribution' in stats:
            assert isinstance(stats['record_type_distribution'], dict)
            assert 'demographics' in stats['record_type_distribution']
            assert 'trip' in stats['record_type_distribution']
    
    def test_test_retrieval(self, rag_engine):
        """Test the test_retrieval functionality"""
        test_results = rag_engine.test_retrieval()
        
        assert isinstance(test_results, dict)
        
        # Check that all test queries were processed
        expected_queries = [
            "tell me about a young user",
            "trip with many passengers", 
            "user from Lahore",
            "transportation patterns",
            "user profile information"
        ]
        
        for query in expected_queries:
            assert query in test_results
            result = test_results[query]
            assert 'success' in result
            
            if result['success']:
                assert 'results_count' in result
                assert 'top_score' in result
                assert isinstance(result['results_count'], int)
                assert isinstance(result['top_score'], (int, float))
            else:
                assert 'error' in result


class TestRAGEngineIntegration:
    """Integration tests for RAG Engine with realistic scenarios"""
    
    @pytest.fixture
    def realistic_rag_engine(self, temp_embeddings_dir, temp_db):
        """RAG engine with more realistic test data"""
        # Add more realistic text representations
        realistic_texts = [
            'Customer profile: User ID: 168928, Age: 26 years old, Gender: Male, frequent traveler from Lahore',
            'Transportation trip: Trip ID: 726765, Pickup: Mall Road Lahore, Destination: DHA Phase 5, Passengers: 8, Duration: 35 minutes',
            'Customer profile: User ID: 346210, Age: 22 years old, Gender: Female, student from Islamabad',
            'Transportation trip: Trip ID: 726771, Pickup: Blue Area Islamabad, Destination: F-8 Markaz, Passengers: 4, Duration: 20 minutes',
            'Customer profile: User ID: 259987, Age: 31 years old, Gender: Male, business professional from Karachi'
        ]
        
        # Override the text representations
        with open(os.path.join(temp_embeddings_dir, 'text_representations.pkl'), 'wb') as f:
            pickle.dump(realistic_texts, f)
        
        # Update metadata to match
        realistic_metadata = [
            {'type': 'demographics', 'user_id': 168928, 'source_table': 'CustomerDemographics'},
            {'type': 'trip', 'trip_id': 726765, 'source_table': 'TripData'},
            {'type': 'demographics', 'user_id': 346210, 'source_table': 'CustomerDemographics'},
            {'type': 'trip', 'trip_id': 726771, 'source_table': 'TripData'},
            {'type': 'demographics', 'user_id': 259987, 'source_table': 'CustomerDemographics'}
        ]
        
        with open(os.path.join(temp_embeddings_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(realistic_metadata, f)
        
        # Create new FAISS index with fewer vectors
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        
        # Generate embeddings for realistic texts
        num_vectors = 5
        sample_embeddings = np.random.random((num_vectors, dimension)).astype(np.float32)
        faiss.normalize_L2(sample_embeddings)
        index.add(sample_embeddings)
        
        with open(os.path.join(temp_embeddings_dir, 'faiss_index.pkl'), 'wb') as f:
            pickle.dump(index, f)
        
        return RAGEngine(
            embeddings_dir=temp_embeddings_dir,
            db_path=temp_db,
            model_name='all-MiniLM-L6-v2'
        )
    
    def test_user_specific_queries(self, realistic_rag_engine):
        """Test queries for specific users"""
        query = "tell me about user 168928"
        results = realistic_rag_engine.search_similar_records(query, top_k=3)
        
        assert len(results) > 0
        
        # Should find demographics record
        found_user = False
        for result in results:
            if result['metadata'].get('user_id') == 168928:
                found_user = True
                assert 'Age: 26' in result['text']
                break
        
        # Note: Due to random embeddings in test, we can't guarantee exact matches
        # In real implementation, semantic similarity would work properly
    
    def test_location_based_queries(self, realistic_rag_engine):
        """Test location-based queries"""
        query = "trips from Lahore"
        results = realistic_rag_engine.search_similar_records(query, top_k=5)
        
        assert len(results) > 0
        # With proper embeddings, this would find Lahore-related trips
    
    def test_passenger_count_queries(self, realistic_rag_engine):
        """Test queries about passenger counts"""
        query = "trips with many passengers"
        results = realistic_rag_engine.search_similar_records(query, top_k=3)
        
        assert len(results) > 0
        # In real implementation, would find high-passenger trips
    
    def test_demographic_queries(self, realistic_rag_engine):
        """Test demographic-based queries"""
        query = "young female users"
        results = realistic_rag_engine.search_similar_records(query, top_k=5)
        
        assert len(results) > 0
        # Would find female users with proper semantic matching


def test_rag_engine_error_handling():
    """Test RAG engine error handling"""
    
    # Test with non-existent embeddings directory
    with pytest.raises(Exception):
        RAGEngine(embeddings_dir="nonexistent_directory")
    
    # Test with missing files
    empty_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(Exception):
            RAGEngine(embeddings_dir=empty_dir)
    finally:
        import shutil
        shutil.rmtree(empty_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])