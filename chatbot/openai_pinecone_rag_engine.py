import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIPineconeRAGEngine:
    """
    RAG (Retrieval-Augmented Generation) Engine using OpenAI embeddings and Pinecone
    """
    
    def __init__(self, 
                 db_path: str = 'database/transportation.db',
                 pinecone_api_key: str = None,
                 openai_api_key: str = None,
                 index_name: str = 'fetii-chatbot',
                 environment: str = 'us-east-1'):
        """
        Initialize RAG Engine with OpenAI and Pinecone
        
        Args:
            db_path: Path to SQLite database for additional context
            pinecone_api_key: Pinecone API key
            openai_api_key: OpenAI API key
            index_name: Pinecone index name
            environment: Pinecone environment
        """
        self.db_path = db_path
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'fetii-chatbot')
        self.environment = environment or os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        
        # Components to be loaded
        self.pc = None
        self.index = None
        self.openai_client = None
        self.config = None
        self.engine = None
        
        # Initialize components
        self.load_pinecone_components()
        self.load_openai_client()
        self.connect_to_database()
        self.load_config()
    
    def load_pinecone_components(self):
        """Load Pinecone client and index"""
        try:
            from pinecone import Pinecone
            
            logger.info("Loading Pinecone components...")
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Connect to existing index
            self.index = self.pc.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index '{self.index_name}' with {stats['total_vector_count']} vectors")
            
        except Exception as e:
            logger.error(f"Error loading Pinecone components: {e}")
            raise
    
    def load_openai_client(self):
        """Load OpenAI client"""
        try:
            from openai import OpenAI
            
            logger.info("Loading OpenAI client...")
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading OpenAI client: {e}")
            raise
    
    def connect_to_database(self):
        """Connect to SQLite database"""
        try:
            if os.path.exists(self.db_path):
                self.engine = create_engine(f'sqlite:///{self.db_path}')
                logger.info(f"Connected to database: {self.db_path}")
            else:
                logger.warning(f"Database file not found: {self.db_path}")
                self.engine = None
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            self.engine = None
    
    def load_config(self):
        """Load configuration if available"""
        try:
            config_path = 'embeddings/openai_pinecone_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("Loaded OpenAI + Pinecone configuration")
            else:
                logger.warning("No configuration file found")
                self.config = {}
        except Exception as e:
            logger.warning(f"Could not load configuration: {e}")
            self.config = {}
    
    def get_openai_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            raise
    
    def search_similar_vectors(self, query: str, top_k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """
        Search for similar vectors in Pinecone
        
        Args:
            query: Search query text
            top_k: Number of results to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            List of matching results with metadata
        """
        try:
            # Get query embedding
            query_embedding = self.get_openai_embedding(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Filter and format results
            filtered_results = []
            for match in results['matches']:
                if match['score'] >= threshold:
                    result = {
                        'score': float(match['score']),
                        'metadata': match['metadata'],
                        'text': match['metadata'].get('text', ''),
                        'id': match['id']
                    }
                    filtered_results.append(result)
            
            logger.info(f"Found {len(filtered_results)} similar vectors for query: '{query[:50]}...'")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def get_database_context(self, metadata_list: List[Dict]) -> List[Dict]:
        """
        Get additional context from database based on retrieved metadata
        
        Args:
            metadata_list: List of metadata from vector search results
            
        Returns:
            List of enhanced context with database information
        """
        if not self.engine:
            return []
        
        enhanced_context = []
        
        try:
            with self.engine.connect() as conn:
                for meta in metadata_list:
                    context = {'original_metadata': meta}
                    
                    # Get user context if user_id is available
                    if 'user_id' in meta and meta['user_id']:
                        user_query = text("""
                            SELECT cd.*, COUNT(ci.TripID) as trip_count
                            FROM CustomerDemographics cd
                            LEFT JOIN CheckedInUsers ci ON cd.UserID = ci.UserID
                            WHERE cd.UserID = :user_id
                            GROUP BY cd.UserID
                        """)
                        user_result = conn.execute(user_query, {'user_id': meta['user_id']}).fetchone()
                        if user_result:
                            context['user_info'] = dict(user_result._mapping)
                    
                    # Get trip context if trip_id is available
                    if 'trip_id' in meta and meta['trip_id']:
                        trip_query = text("""
                            SELECT td.*, COUNT(ci.UserID) as passenger_count
                            FROM TripData td
                            LEFT JOIN CheckedInUsers ci ON td.TripID = ci.TripID
                            WHERE td.TripID = :trip_id
                            GROUP BY td.TripID
                        """)
                        trip_result = conn.execute(trip_query, {'trip_id': meta['trip_id']}).fetchone()
                        if trip_result:
                            context['trip_info'] = dict(trip_result._mapping)
                    
                    enhanced_context.append(context)
                    
        except Exception as e:
            logger.error(f"Error getting database context: {e}")
        
        return enhanced_context
    
    def format_context_snippets(self, results: List[Dict], enhanced_context: List[Dict]) -> List[Dict]:
        """
        Format search results into context snippets for RAG
        
        Args:
            results: Raw search results from Pinecone
            enhanced_context: Enhanced context from database
            
        Returns:
            Formatted context snippets
        """
        context_snippets = []
        
        for i, (result, enhanced) in enumerate(zip(results, enhanced_context)):
            snippet = {
                'rank': i + 1,
                'relevance_score': result['score'],
                'content': result['text'],
                'type': result['metadata'].get('type', 'unknown'),
                'source': 'vector_search',
                'identifiers': {
                    'id': result['id'],
                    'user_id': result['metadata'].get('user_id'),
                    'trip_id': result['metadata'].get('trip_id')
                }
            }
            
            # Add enhanced database context
            if 'user_info' in enhanced:
                snippet['user_context'] = enhanced['user_info']
            
            if 'trip_info' in enhanced:
                snippet['trip_context'] = enhanced['trip_info']
            
            context_snippets.append(snippet)
        
        return context_snippets
    
    def process_rag_query(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Main RAG processing function
        
        Args:
            query: Natural language query
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            Dictionary containing RAG processing results
        """
        start_time = datetime.now()
        
        try:
            # Search for similar vectors
            search_results = self.search_similar_vectors(
                query=query,
                top_k=top_k,
                threshold=similarity_threshold
            )
            
            if not search_results:
                return {
                    'success': False,
                    'error': 'No relevant context found',
                    'context_snippets': [],
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Get enhanced database context
            enhanced_context = self.get_database_context([r['metadata'] for r in search_results])
            
            # Format context snippets
            context_snippets = self.format_context_snippets(search_results, enhanced_context)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"RAG query processed in {processing_time:.3f}s, found {len(context_snippets)} context snippets")
            
            return {
                'success': True,
                'query': query,
                'context_snippets': context_snippets,
                'total_results': len(context_snippets),
                'processing_time': processing_time,
                'vector_db': 'pinecone',
                'embedding_model': 'text-embedding-3-small'
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error processing RAG query: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'context_snippets': [],
                'processing_time': processing_time
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            
            return {
                'pinecone_connected': True,
                'openai_connected': True,
                'database_connected': self.engine is not None,
                'index_name': self.index_name,
                'total_vectors': stats['total_vector_count'],
                'embedding_model': 'text-embedding-3-small',
                'config_loaded': bool(self.config)
            }
            
        except Exception as e:
            return {
                'pinecone_connected': False,
                'openai_connected': False,
                'database_connected': False,
                'error': str(e)
            }
    
    def search_by_filters(self, filters: Dict[str, Any], top_k: int = 10) -> List[Dict]:
        """
        Search vectors by metadata filters
        
        Args:
            filters: Dictionary of metadata filters
            top_k: Number of results to return
            
        Returns:
            List of matching results
        """
        try:
            # Query with filters only (no vector)
            results = self.index.query(
                vector=[0.0] * 1536,  # Dummy vector
                top_k=top_k,
                include_metadata=True,
                filter=filters
            )
            
            formatted_results = []
            for match in results['matches']:
                result = {
                    'metadata': match['metadata'],
                    'text': match['metadata'].get('text', ''),
                    'id': match['id']
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return []

def test_openai_pinecone_rag():
    """Test function for the OpenAI + Pinecone RAG engine"""
    try:
        # Initialize RAG engine
        rag_engine = OpenAIPineconeRAGEngine()
        
        # Test system status
        status = rag_engine.get_system_status()
        print("System Status:", status)
        
        # Test queries
        test_queries = [
            "young male user from Austin",
            "trip with 10 passengers",
            "user aged 25 years",
            "trips to downtown Austin",
            "large group transportation"
        ]
        
        for query in test_queries:
            print(f"\n--- Testing Query: '{query}' ---")
            result = rag_engine.process_rag_query(query, top_k=3)
            
            if result['success']:
                print(f"Found {result['total_results']} results in {result['processing_time']:.3f}s")
                for snippet in result['context_snippets']:
                    print(f"  {snippet['rank']}. Score: {snippet['relevance_score']:.4f}")
                    print(f"     Type: {snippet['type']}")
                    print(f"     Content: {snippet['content'][:100]}...")
            else:
                print(f"Error: {result['error']}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_openai_pinecone_rag()
