import chromadb
from chromadb.config import Settings
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) Engine for semantic search and retrieval
    """
    
    def __init__(self, 
                 chroma_persist_dir: str = 'embeddings/chromadb',
                 db_path: str = 'database/transportation.db',
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize RAG Engine
        
        Args:
            chroma_persist_dir: Directory containing ChromaDB persistent storage
            db_path: Path to SQLite database for additional context
            model_name: SentenceTransformer model name
        """
        self.chroma_persist_dir = chroma_persist_dir
        self.db_path = db_path
        self.model_name = model_name
        
        # Components to be loaded
        self.chroma_client = None
        self.collection = None
        self.model = None
        self.config = None
        self.engine = None
        
        # Initialize components
        self.load_chromadb_components()
        self.load_sentence_transformer()
        self.connect_to_database()
    
    def load_chromadb_components(self):
        """Load ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get collection
            collection_name = "transportation_embeddings"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                count = self.collection.count()
                logger.info(f"Loaded ChromaDB collection '{collection_name}' with {count} documents")
            except ValueError:
                raise FileNotFoundError(f"ChromaDB collection '{collection_name}' not found at {self.chroma_persist_dir}")
            
            # Load configuration
            config_path = os.path.join(self.chroma_persist_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("Loaded ChromaDB configuration")
            else:
                logger.warning("ChromaDB configuration not found")
            
        except Exception as e:
            logger.error(f"Error loading ChromaDB components: {e}")
            raise
    
    def load_sentence_transformer(self):
        """Load SentenceTransformer model"""
        try:
            # Force CPU-only to avoid meta-tensor device issues on some setups
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            logger.info(f"Loading SentenceTransformer model on CPU: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device='cpu')
            logger.info("SentenceTransformer model loaded successfully (CPU)")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {e}")
            # Retry with a smaller fallback model
            try:
                fallback_model = "all-MiniLM-L6-v2"
                if self.model_name != fallback_model:
                    logger.info(f"Retrying with fallback model on CPU: {fallback_model}")
                    self.model = SentenceTransformer(fallback_model, device='cpu')
                    self.model_name = fallback_model
                    logger.info("Fallback SentenceTransformer model loaded successfully (CPU)")
                else:
                    raise
            except Exception as e2:
                logger.error(f"Fallback model load failed: {e2}")
                raise
    
    def connect_to_database(self):
        """Connect to database for additional context retrieval"""
        try:
            self.engine = create_engine(f'sqlite:///{self.db_path}')
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            # Don't raise here - RAG can work without DB connection
    
    def encode_query(self, query: str) -> List[float]:
        """
        Encode user query into embedding vector
        
        Args:
            query: User's natural language query
            
        Returns:
            Query embedding vector as list
        """
        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Return as list for ChromaDB
            return query_embedding[0].tolist()
            
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            raise
    
    def search_similar_records(self, query: str, top_k: int = 5, 
                             score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search
        
        Args:
            query: User's natural language query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Search ChromaDB collection
            search_results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Process results
            results = []
            if search_results['documents'] and search_results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    search_results['documents'][0],
                    search_results['metadatas'][0],
                    search_results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    # Check score threshold
                    if similarity_score < score_threshold:
                        continue
                    
                    result = {
                        'rank': i + 1,
                        'score': float(similarity_score),
                        'text': doc,
                        'metadata': metadata.copy(),
                        'query': query
                    }
                    
                    results.append(result)
            
            logger.info(f"Found {len(results)} relevant records for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    def enrich_results_with_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich search results with additional context from database
        
        Args:
            results: List of search results
            
        Returns:
            Enriched results with additional context
        """
        if not self.engine:
            logger.warning("Database connection not available for context enrichment")
            return results
        
        enriched_results = []
        
        for result in results:
            enriched_result = result.copy()
            metadata = result['metadata']
            
            try:
                with self.engine.connect() as conn:
                    additional_context = {}
                    
                    # Get additional user context if UserID available
                    if 'user_id' in metadata and metadata['user_id']:
                        user_query = text("""
                            SELECT cd.*, COUNT(ci.TripID) as total_trips
                            FROM CustomerDemographics cd
                            LEFT JOIN CheckedInUsers ci ON cd.UserID = ci.UserID
                            WHERE cd.UserID = :user_id
                            GROUP BY cd.UserID, cd.Age, cd.Gender, cd.Name
                        """)
                        user_result = conn.execute(user_query, {'user_id': metadata['user_id']}).fetchone()
                        
                        if user_result:
                            additional_context['user_details'] = dict(user_result._mapping)
                    
                    # Get additional trip context if TripID available
                    if 'trip_id' in metadata and metadata['trip_id']:
                        trip_query = text("""
                            SELECT t.*, COUNT(ci.UserID) as checked_in_users
                            FROM TripData t
                            LEFT JOIN CheckedInUsers ci ON t.TripID = ci.TripID
                            WHERE t.TripID = :trip_id
                            GROUP BY t.TripID
                        """)
                        trip_result = conn.execute(trip_query, {'trip_id': metadata['trip_id']}).fetchone()
                        
                        if trip_result:
                            additional_context['trip_details'] = dict(trip_result._mapping)
                    
                    # Get related users for trip-based results
                    if metadata.get('type') == 'trip' and 'trip_id' in metadata:
                        related_users_query = text("""
                            SELECT cd.UserID, cd.Name, cd.Age, cd.Gender
                            FROM CustomerDemographics cd
                            JOIN CheckedInUsers ci ON cd.UserID = ci.UserID
                            WHERE ci.TripID = :trip_id
                        """)
                        related_users = conn.execute(related_users_query, 
                                                   {'trip_id': metadata['trip_id']}).fetchall()
                        
                        if related_users:
                            additional_context['related_users'] = [dict(row._mapping) for row in related_users]
                    
                    # Add enriched context
                    enriched_result['additional_context'] = additional_context
                    
            except Exception as e:
                logger.warning(f"Error enriching result context: {e}")
                enriched_result['additional_context'] = {}
            
            enriched_results.append(enriched_result)
        
        return enriched_results
    
    def format_results_for_llm(self, results: List[Dict[str, Any]], 
                              query: str, max_context_length: int = 2000) -> Dict[str, Any]:
        """
        Format search results for LLM consumption
        
        Args:
            results: Search results
            query: Original query
            max_context_length: Maximum total context length
            
        Returns:
            Formatted context for LLM
        """
        try:
            # Sort results by score (descending)
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Format context snippets
            context_snippets = []
            total_length = 0
            
            for i, result in enumerate(sorted_results):
                # Create context snippet
                snippet = {
                    'rank': i + 1,
                    'relevance_score': result['score'],
                    'content': result['text'],
                    'type': result['metadata'].get('type', 'unknown'),
                    'source': result['metadata'].get('source_table', 'unknown')
                }
                
                # Add key identifiers
                metadata = result['metadata']
                identifiers = []
                
                if 'user_id' in metadata and metadata['user_id']:
                    identifiers.append(f"UserID: {metadata['user_id']}")
                
                if 'trip_id' in metadata and metadata['trip_id']:
                    identifiers.append(f"TripID: {metadata['trip_id']}")
                
                if identifiers:
                    snippet['identifiers'] = ', '.join(identifiers)
                
                # Add additional context if available
                if 'additional_context' in result and result['additional_context']:
                    context_summary = []
                    
                    if 'user_details' in result['additional_context']:
                        user_det = result['additional_context']['user_details']
                        context_summary.append(f"User has {user_det.get('total_trips', 0)} total trips")
                    
                    if 'trip_details' in result['additional_context']:
                        trip_det = result['additional_context']['trip_details']
                        context_summary.append(f"Trip has {trip_det.get('checked_in_users', 0)} checked-in users")
                    
                    if 'related_users' in result['additional_context']:
                        related_count = len(result['additional_context']['related_users'])
                        context_summary.append(f"{related_count} related users")
                    
                    if context_summary:
                        snippet['additional_info'] = '; '.join(context_summary)
                
                # Check length constraint
                snippet_text = json.dumps(snippet)
                if total_length + len(snippet_text) <= max_context_length:
                    context_snippets.append(snippet)
                    total_length += len(snippet_text)
                else:
                    logger.info(f"Truncated context at {len(context_snippets)} snippets due to length constraint")
                    break
            
            # Create formatted response
            formatted_response = {
                'query': query,
                'total_results': len(results),
                'returned_results': len(context_snippets),
                'context_snippets': context_snippets,
                'search_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_used': self.model_name,
                    'embedding_dimension': self.config.get('embedding_dimension') if self.config else None,
                    'total_vectors_searched': self.collection.count() if self.collection else 0
                }
            }
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting results for LLM: {e}")
            raise
    
    def retrieve_context(self, query: str, top_k: int = 5, 
                        score_threshold: float = 0.1, 
                        enrich_context: bool = True,
                        max_context_length: int = 2000) -> Dict[str, Any]:
        """
        Complete RAG retrieval pipeline
        
        Args:
            query: User's natural language query
            top_k: Number of top results to retrieve
            score_threshold: Minimum similarity score
            enrich_context: Whether to enrich with additional database context
            max_context_length: Maximum context length for LLM
            
        Returns:
            Formatted context ready for LLM consumption
        """
        try:
            logger.info(f"Processing RAG query: '{query}'")
            
            # Step 1: Semantic search
            search_results = self.search_similar_records(
                query=query, 
                top_k=top_k, 
                score_threshold=score_threshold
            )
            
            if not search_results:
                return {
                    'query': query,
                    'total_results': 0,
                    'returned_results': 0,
                    'context_snippets': [],
                    'message': 'No relevant records found for the query'
                }
            
            # Step 2: Enrich with additional context
            if enrich_context:
                search_results = self.enrich_results_with_context(search_results)
            
            # Step 3: Format for LLM
            formatted_context = self.format_results_for_llm(
                results=search_results,
                query=query,
                max_context_length=max_context_length
            )
            
            logger.info(f"Retrieved {formatted_context['returned_results']} context snippets")
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error in RAG retrieval pipeline: {e}")
            return {
                'query': query,
                'error': str(e),
                'total_results': 0,
                'returned_results': 0,
                'context_snippets': []
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        try:
            collection_count = self.collection.count() if self.collection else 0
            
            stats = {
                'chromadb_collection_size': collection_count,
                'model_name': self.model_name,
                'chroma_persist_dir': self.chroma_persist_dir,
                'collection_name': self.collection.name if self.collection else 'unknown'
            }
            
            # Add configuration info
            if self.config:
                stats['total_documents'] = self.config.get('total_documents', 0)
                stats['creation_timestamp'] = self.config.get('creation_timestamp')
                stats['db_path'] = self.config.get('db_path')
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    def test_retrieval(self) -> Dict[str, Any]:
        """Test RAG retrieval with sample queries"""
        test_queries = [
            "tell me about a young user",
            "trip with many passengers",
            "user from Lahore",
            "transportation patterns",
            "user profile information"
        ]
        
        test_results = {}
        
        for query in test_queries:
            try:
                result = self.retrieve_context(query, top_k=3)
                test_results[query] = {
                    'success': True,
                    'results_count': result['returned_results'],
                    'top_score': result['context_snippets'][0]['relevance_score'] if result['context_snippets'] else 0
                }
            except Exception as e:
                test_results[query] = {
                    'success': False,
                    'error': str(e)
                }
        
        return test_results

def main():
    """Test RAG Engine functionality"""
    print("Testing RAG Engine")
    print("=" * 50)
    
    try:
        # Initialize RAG Engine
        rag_engine = RAGEngine()
        
        # Get statistics
        stats = rag_engine.get_statistics()
        print("RAG Engine Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n{'='*50}")
        print("Testing Semantic Retrieval:")
        print("-" * 50)
        
        # Test queries
        test_queries = [
            "tell me about user 168928",
            "show trips with more than 8 passengers",
            "young male users from the dataset",
            "transportation patterns in Lahore"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            # Retrieve context
            result = rag_engine.retrieve_context(query, top_k=3)
            
            if result['returned_results'] > 0:
                print(f"   Found {result['returned_results']} relevant records:")
                
                for snippet in result['context_snippets']:
                    print(f"   - Score: {snippet['relevance_score']:.4f} | {snippet['type']}")
                    print(f"     {snippet['content'][:100]}...")
                    if 'identifiers' in snippet:
                        print(f"     {snippet['identifiers']}")
            else:
                print(f"   No relevant records found")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
        
        print(f"\n{'='*50}")
        print("Running System Test:")
        print("-" * 50)
        
        test_results = rag_engine.test_retrieval()
        for query, result in test_results.items():
            status = "✓" if result['success'] else "✗"
            print(f"{status} {query}")
            if result['success']:
                print(f"   Results: {result['results_count']}, Top Score: {result['top_score']:.4f}")
            else:
                print(f"   Error: {result['error']}")
    
    except Exception as e:
        print(f"Error initializing RAG Engine: {e}")
        print("Make sure you have run build_embeddings.py first to create the ChromaDB collection")

if __name__ == "__main__":
    main()