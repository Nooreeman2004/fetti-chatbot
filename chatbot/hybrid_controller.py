import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import traceback

# Import your existing components
from .query_classifier import QueryClassifier, QueryType
from .sql_engine import SQLEngine
from .openai_pinecone_rag_engine import OpenAIPineconeRAGEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridController:
    """
    Hybrid Controller Layer - Orchestrates SQL, RAG, and Hybrid query processing
    """
    
    def __init__(self, 
                 db_path: str = 'database/transportation.db',
                 pinecone_api_key: str = None,
                 openai_api_key: str = None,
                 pinecone_index_name: str = 'fetii-chatbot',
                 pinecone_environment: str = 'us-east-1',
                 default_top_k: int = 5,
                 confidence_threshold: float = 0.5):
        """
        Initialize Hybrid Controller with all engines
        
        Args:
            db_path: Path to SQLite database
            pinecone_api_key: Pinecone API key
            openai_api_key: OpenAI API key for embeddings, NL-to-SQL and classification
            pinecone_index_name: Pinecone index name
            pinecone_environment: Pinecone environment
            default_top_k: Default number of RAG results to retrieve
            confidence_threshold: Minimum confidence for query classification
        """
        self.db_path = db_path
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_environment = pinecone_environment
        self.default_top_k = default_top_k
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.query_classifier = None
        self.sql_engine = None
        self.rag_engine = None
        
        # Track initialization status
        self.initialization_status = {}
        
        # Initialize all components
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all engine components"""
        logger.info("Initializing Hybrid Controller components...")
        
        # Initialize Query Classifier
        try:
            self.query_classifier = QueryClassifier(openai_api_key=self.openai_api_key)
            self.initialization_status['query_classifier'] = {'status': 'success', 'error': None}
            logger.info("Query Classifier initialized successfully")
        except Exception as e:
            self.initialization_status['query_classifier'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"Failed to initialize Query Classifier: {e}")
        
        # Initialize SQL Engine
        try:
            self.sql_engine = SQLEngine(db_path=self.db_path, openai_api_key=self.openai_api_key)
            self.initialization_status['sql_engine'] = {'status': 'success', 'error': None}
            logger.info("SQL Engine initialized successfully")
        except Exception as e:
            self.initialization_status['sql_engine'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"Failed to initialize SQL Engine: {e}")
        
        # Initialize RAG Engine
        try:
            self.rag_engine = OpenAIPineconeRAGEngine(
                db_path=self.db_path,
                pinecone_api_key=self.pinecone_api_key,
                openai_api_key=self.openai_api_key,
                index_name=self.pinecone_index_name,
                environment=self.pinecone_environment
            )
            self.initialization_status['rag_engine'] = {'status': 'success', 'error': None}
            logger.info("OpenAI + Pinecone RAG Engine initialized successfully")
        except Exception as e:
            self.initialization_status['rag_engine'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"Failed to initialize RAG Engine: {e}")
        
        # Log overall initialization status
        successful_components = sum(1 for status in self.initialization_status.values() 
                                  if status['status'] == 'success')
        total_components = len(self.initialization_status)
        
        logger.info(f"Hybrid Controller initialization: {successful_components}/{total_components} components successful")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components"""
        return {
            'timestamp': datetime.now().isoformat(),
            'components': self.initialization_status,
            'overall_status': 'healthy' if all(
                status['status'] == 'success' 
                for status in self.initialization_status.values()
            ) else 'degraded'
        }
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify user query using the Query Classifier
        
        Args:
            query: User's natural language query
            
        Returns:
            Classification result with confidence and method info
        """
        if not self.query_classifier:
            return {
                'classification': None,
                'confidence': 0.0,
                'method': 'classifier_unavailable',
                'error': 'Query Classifier not initialized'
            }
        
        try:
            return self.query_classifier.classify_query(query, self.confidence_threshold)
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            return {
                'classification': None,
                'confidence': 0.0,
                'method': 'classification_error',
                'error': str(e)
            }
    
    def process_sql_query(self, query: str) -> Dict[str, Any]:
        """
        Process query using SQL Engine
        
        Args:
            query: User's natural language query
            
        Returns:
            SQL processing results
        """
        if not self.sql_engine:
            return {
                'success': False,
                'error': 'SQL Engine not initialized',
                'data': None,
                'query_type': 'SQL'
            }
        
        try:
            result = self.sql_engine.process_natural_language_query(query)
            result['query_type'] = 'SQL'
            
            # Enhance the query if it was successful
            if result.get('success') and result.get('sql_query'):
                enhancement = self.sql_engine.validate_and_enhance_query(
                    result['sql_query'], query
                )
                if enhancement.get('valid'):
                    result['enhanced_query'] = enhancement['enhanced_query']
                    result['query_improvements'] = enhancement.get('improvements', [])
            
            return result
        except Exception as e:
            logger.error(f"Error in SQL processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None,
                'query_type': 'SQL',
                'natural_query': query
            }
    
    def process_rag_query(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Process query using RAG Engine
        
        Args:
            query: User's natural language query
            top_k: Number of results to retrieve
            
        Returns:
            RAG processing results
        """
        if not self.rag_engine:
            return {
                'success': False,
                'error': 'RAG Engine not initialized',
                'context_snippets': [],
                'query_type': 'RAG'
            }
        
        if top_k is None:
            top_k = self.default_top_k
        
        try:
            result = self.rag_engine.retrieve_context(query, top_k=top_k)
            
            # Format for consistent interface
            return {
                'success': result['returned_results'] > 0,
                'query': result['query'],
                'context_snippets': result['context_snippets'],
                'total_results': result['total_results'],
                'returned_results': result['returned_results'],
                'search_metadata': result.get('search_metadata', {}),
                'query_type': 'RAG',
                'error': result.get('error')
            }
        except Exception as e:
            logger.error(f"Error in RAG processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'context_snippets': [],
                'query_type': 'RAG',
                'query': query
            }
    
    def process_hybrid_query(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Process query using both SQL and RAG engines for hybrid approach
        
        Args:
            query: User's natural language query
            top_k: Number of RAG results to retrieve
            
        Returns:
            Combined SQL and RAG results
        """
        if top_k is None:
            top_k = self.default_top_k
        
        logger.info(f"Processing hybrid query: '{query}'")
        
        hybrid_result = {
            'success': False,
            'query': query,
            'query_type': 'HYBRID',
            'sql_result': None,
            'rag_result': None,
            'combined_insights': [],
            'error': None
        }
        
        try:
            # Step 1: Process with SQL Engine
            logger.info("Step 1: Processing with SQL Engine")
            sql_result = self.process_sql_query(query)
            hybrid_result['sql_result'] = sql_result
            
            # Step 2: Process with RAG Engine
            logger.info("Step 2: Processing with RAG Engine")
            rag_result = self.process_rag_query(query, top_k)
            hybrid_result['rag_result'] = rag_result
            
            # Step 3: Combine and enhance results
            logger.info("Step 3: Combining SQL and RAG results")
            
            # Check if we have SQL results to enhance RAG search
            if sql_result['success'] and hasattr(sql_result.get('data'), 'iterrows'):
                # Extract key entities from SQL results for enhanced RAG search
                enhanced_rag_queries = self._extract_entities_for_rag_enhancement(sql_result['data'])
                
                # Perform additional RAG searches based on SQL results
                enhanced_contexts = []
                for enhanced_query in enhanced_rag_queries[:3]:  # Limit to top 3 enhanced queries
                    enhanced_rag = self.process_rag_query(enhanced_query, top_k=3)
                    if enhanced_rag['success']:
                        enhanced_contexts.extend(enhanced_rag['context_snippets'])
                
                # Add enhanced contexts to main RAG result
                if enhanced_contexts:
                    hybrid_result['rag_result']['context_snippets'].extend(enhanced_contexts)
                    hybrid_result['rag_result']['returned_results'] += len(enhanced_contexts)
            
            # Step 4: Generate combined insights
            combined_insights = self._generate_combined_insights(
                sql_result, 
                hybrid_result['rag_result']
            )
            hybrid_result['combined_insights'] = combined_insights
            
            # Determine overall success
            hybrid_result['success'] = (
                sql_result.get('success', False) or 
                hybrid_result['rag_result'].get('success', False)
            )
            
            if not hybrid_result['success']:
                hybrid_result['error'] = 'Both SQL and RAG processing failed'
            
            logger.info(f"Hybrid processing completed. Success: {hybrid_result['success']}")
            return hybrid_result
            
        except Exception as e:
            logger.error(f"Error in hybrid processing: {e}")
            hybrid_result['error'] = str(e)
            return hybrid_result
    
    def _extract_entities_for_rag_enhancement(self, sql_data) -> List[str]:
        """
        Extract entities from SQL results to enhance RAG search
        
        Args:
            sql_data: DataFrame or dict from SQL results
            
        Returns:
            List of enhanced query strings for RAG
        """
        enhanced_queries = []
        
        try:
            if hasattr(sql_data, 'iterrows'):  # DataFrame
                # Extract specific user IDs, trip IDs for detailed RAG search
                if 'UserID' in sql_data.columns:
                    for user_id in sql_data['UserID'].head(3):  # Top 3 users
                        enhanced_queries.append(f"user {user_id} profile details")
                
                if 'TripID' in sql_data.columns:
                    for trip_id in sql_data['TripID'].head(3):  # Top 3 trips
                        enhanced_queries.append(f"trip {trip_id} details")
                
                # Extract location-based queries
                if 'PickUpAddress' in sql_data.columns:
                    for address in sql_data['PickUpAddress'].dropna().head(2):
                        if address and not address.startswith('#'):
                            enhanced_queries.append(f"trips from {address[:50]}")
                
        except Exception as e:
            logger.warning(f"Error extracting entities for RAG enhancement: {e}")
        
        return enhanced_queries
    
    def _generate_combined_insights(self, sql_result: Dict, rag_result: Dict) -> List[Dict[str, Any]]:
        """
        Generate insights by combining SQL and RAG results
        
        Args:
            sql_result: SQL processing result
            rag_result: RAG processing result
            
        Returns:
            List of combined insights
        """
        insights = []
        
        try:
            # SQL Data Summary
            if sql_result.get('success') and sql_result.get('data') is not None:
                if hasattr(sql_result['data'], 'shape'):
                    insights.append({
                        'type': 'sql_summary',
                        'title': 'Structured Data Analysis',
                        'content': f"Found {sql_result['data'].shape[0]} records with {sql_result['data'].shape[1]} attributes",
                        'metadata': {
                            'row_count': sql_result['data'].shape[0],
                            'execution_time': sql_result.get('execution_time', 0)
                        }
                    })
            
            # RAG Context Summary
            if rag_result.get('success') and rag_result.get('context_snippets'):
                top_score = max(snippet['relevance_score'] for snippet in rag_result['context_snippets'])
                insights.append({
                    'type': 'rag_summary',
                    'title': 'Contextual Information',
                    'content': f"Retrieved {len(rag_result['context_snippets'])} relevant context snippets (top relevance: {top_score:.3f})",
                    'metadata': {
                        'snippet_count': len(rag_result['context_snippets']),
                        'top_relevance_score': top_score
                    }
                })
            
            # Cross-reference insights
            if (sql_result.get('success') and rag_result.get('success') and 
                hasattr(sql_result.get('data'), 'iterrows') and rag_result.get('context_snippets')):
                
                # Find overlapping entities
                sql_entities = self._extract_entities_from_sql(sql_result['data'])
                rag_entities = self._extract_entities_from_rag(rag_result['context_snippets'])
                
                overlap = set(sql_entities) & set(rag_entities)
                if overlap:
                    insights.append({
                        'type': 'cross_reference',
                        'title': 'Cross-Referenced Entities',
                        'content': f"Found {len(overlap)} entities present in both structured data and context",
                        'metadata': {
                            'overlapping_entities': list(overlap)[:5]  # Top 5
                        }
                    })
            
        except Exception as e:
            logger.warning(f"Error generating combined insights: {e}")
            insights.append({
                'type': 'error',
                'title': 'Insight Generation Error',
                'content': f"Could not generate combined insights: {str(e)}"
            })
        
        return insights
    
    def _extract_entities_from_sql(self, data) -> List[str]:
        """Extract entity identifiers from SQL results"""
        entities = []
        try:
            if 'UserID' in data.columns:
                entities.extend([f"user_{uid}" for uid in data['UserID'].head(5)])
            if 'TripID' in data.columns:
                entities.extend([f"trip_{tid}" for tid in data['TripID'].head(5)])
        except Exception:
            pass
        return entities
    
    def _extract_entities_from_rag(self, context_snippets: List[Dict]) -> List[str]:
        """Extract entity identifiers from RAG results"""
        entities = []
        try:
            for snippet in context_snippets:
                metadata = snippet.get('metadata', {})
                if metadata.get('user_id'):
                    entities.append(f"user_{metadata['user_id']}")
                if metadata.get('trip_id'):
                    entities.append(f"trip_{metadata['trip_id']}")
        except Exception:
            pass
        return entities
    
    def process_query(self, query: str, force_type: str = None, **kwargs) -> Dict[str, Any]:
        """
        Main query processing method - orchestrates the entire pipeline
        
        Args:
            query: User's natural language query
            force_type: Force specific processing type ('SQL', 'RAG', 'HYBRID')
            **kwargs: Additional arguments (top_k, etc.)
            
        Returns:
            Complete processing results
        """
        start_time = datetime.now()
        
        # Create base result structure
        result = {
            'query': query,
            'timestamp': start_time.isoformat(),
            'classification': None,
            'processing_type': None,
            'success': False,
            'error': None,
            'processing_time': 0,
            'results': {}
        }
        
        try:
            logger.info(f"Processing query: '{query}'")
            
            # Step 1: Query Classification (unless forced)
            if force_type:
                logger.info(f"Forced processing type: {force_type}")
                result['classification'] = {
                    'classification': QueryType(force_type),
                    'confidence': 1.0,
                    'method': 'forced'
                }
            else:
                classification_result = self.classify_query(query)
                result['classification'] = classification_result
                
                if not classification_result.get('classification'):
                    result['error'] = 'Query classification failed'
                    result['processing_time'] = (datetime.now() - start_time).total_seconds()
                    return result
            
            # Step 2: Route to appropriate engine(s)
            query_type = (result['classification']['classification'].value 
                         if result['classification']['classification'] else 'RAG')
            result['processing_type'] = query_type
            
            if query_type == 'SQL':
                logger.info("Routing to SQL Engine")
                processing_result = self.process_sql_query(query)
                result['results'] = processing_result
                result['success'] = processing_result.get('success', False)
                
            elif query_type == 'RAG':
                logger.info("Routing to RAG Engine")
                processing_result = self.process_rag_query(query, kwargs.get('top_k'))
                result['results'] = processing_result
                result['success'] = processing_result.get('success', False)
                
            elif query_type == 'HYBRID':
                logger.info("Routing to Hybrid Processing")
                processing_result = self.process_hybrid_query(query, kwargs.get('top_k'))
                result['results'] = processing_result
                result['success'] = processing_result.get('success', False)
                
            elif query_type == 'OFF_TOPIC':
                logger.info("Query classified as OFF_TOPIC - rejecting")
                result['success'] = True  # Successfully handled by rejecting
                result['results'] = {
                    'success': True,
                    'message': 'off_topic_query',
                    'query_type': 'OFF_TOPIC'
                }
            
            else:
                result['error'] = f'Unknown query type: {query_type}'
            
            # Step 3: Set error if processing failed
            if not result['success'] and not result['error']:
                result['error'] = result['results'].get('error', 'Unknown processing error')
            
        except Exception as e:
            logger.error(f"Error in query processing pipeline: {e}")
            logger.error(traceback.format_exc())
            result['error'] = str(e)
            result['success'] = False
        
        # Calculate processing time
        result['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Query processing completed. Success: {result['success']}, "
                   f"Time: {result['processing_time']:.3f}s")
        
        return result

def main():
    """Test the Hybrid Controller functionality"""
    print("Testing Hybrid Controller")
    print("=" * 60)
    
    # Initialize controller
    try:
        controller = HybridController()
    except Exception as e:
        print(f"Failed to initialize Hybrid Controller: {e}")
        return
    
    # Check system status
    status = controller.get_system_status()
    print("System Status:")
    print(f"  Overall: {status['overall_status']}")
    for component, info in status['components'].items():
        print(f"  {component}: {info['status']}")
        if info['error']:
            print(f"    Error: {info['error']}")
    
    print(f"\n{'='*60}")
    print("Testing Query Processing Pipeline")
    print("-" * 60)
    
    # Test queries for different types
    test_queries = [
        ("How many users are in the database?", "SQL"),
        ("Tell me about user 168928", "RAG"), 
        ("Show me the top 5 users by trip count and their profiles", "HYBRID"),
        ("What is the average trip duration?", None),  # Let classifier decide
    ]
    
    for i, (query, force_type) in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        if force_type:
            print(f"   Forced Type: {force_type}")
        
        # Process query
        result = controller.process_query(query, force_type=force_type)
        
        print(f"   Classification: {result['classification']['classification'].value if result['classification'] and result['classification']['classification'] else 'None'}")
        print(f"   Processing Type: {result['processing_type']}")
        print(f"   Success: {result['success']}")
        print(f"   Processing Time: {result['processing_time']:.3f}s")
        
        if result['success']:
            if result['processing_type'] == 'SQL':
                sql_results = result['results']
                if hasattr(sql_results.get('data'), 'shape'):
                    print(f"   SQL Results: {sql_results['data'].shape[0]} rows")
            
            elif result['processing_type'] == 'RAG':
                rag_results = result['results']
                print(f"   RAG Results: {rag_results.get('returned_results', 0)} snippets")
            
            elif result['processing_type'] == 'HYBRID':
                hybrid_results = result['results']
                sql_success = hybrid_results.get('sql_result', {}).get('success', False)
                rag_success = hybrid_results.get('rag_result', {}).get('success', False)
                print(f"   Hybrid Results: SQL={sql_success}, RAG={rag_success}")
                print(f"   Combined Insights: {len(hybrid_results.get('combined_insights', []))}")
        else:
            print(f"   Error: {result['error']}")
    
    print(f"\n{'='*60}")
    print("Hybrid Controller Testing Complete")

if __name__ == "__main__":
    main()