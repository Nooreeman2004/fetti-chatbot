import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import traceback

# Import your existing components
from .query_classifier import QueryClassifier, QueryType
from .sql_engine import SQLEngine
from .openai_pinecone_rag_engine import OpenAIPineconeRAGEngine
from .response_generator import ResponseGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridController:
    """
    Hybrid Controller Layer - Orchestrates SQL, RAG, and Response Generation
    """
    
    def __init__(self, 
                 db_path: str = 'database/transportation.db',
                 embeddings_dir: str = 'embeddings/vectordb',
                 openai_api_key: str = None,
                 default_top_k: int = 5,
                 confidence_threshold: float = 0.6):
        """
        Initialize Hybrid Controller with all engines
        
        Args:
            db_path: Path to SQLite database
            embeddings_dir: Directory containing FAISS embeddings
            openai_api_key: OpenAI API key for NL-to-SQL and classification
            default_top_k: Default number of RAG results to retrieve
            confidence_threshold: Minimum confidence for query classification
        """
        self.db_path = db_path
        self.embeddings_dir = embeddings_dir
        self.openai_api_key = openai_api_key
        self.default_top_k = default_top_k
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.query_classifier = None
        self.sql_engine = None
        self.rag_engine = None
        self.response_generator = None
        
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
        
        # Initialize RAG Engine (optional - system can work without it)
        try:
            self.rag_engine = OpenAIPineconeRAGEngine(db_path=self.db_path)
            self.initialization_status['rag_engine'] = {'status': 'success', 'error': None}
            logger.info("Pinecone RAG Engine initialized successfully")
        except Exception as e:
            self.initialization_status['rag_engine'] = {'status': 'failed', 'error': str(e)}
            logger.warning(f"RAG Engine initialization failed: {e}. System will work with SQL-only mode.")
        
        # Initialize Response Generator
        try:
            self.response_generator = ResponseGenerator(openai_api_key=self.openai_api_key)
            self.initialization_status['response_generator'] = {'status': 'success', 'error': None}
            logger.info("Response Generator initialized successfully")
        except Exception as e:
            self.initialization_status['response_generator'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"Failed to initialize Response Generator: {e}")
        
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
            ) else 'degraded',
            'core_functionality': self._assess_core_functionality()
        }
    
    def _assess_core_functionality(self) -> Dict[str, bool]:
        """Assess which core functionalities are available"""
        return {
            'sql_queries': self.sql_engine is not None,
            'rag_search': self.rag_engine is not None,
            'query_classification': self.query_classifier is not None,
            'response_generation': self.response_generator is not None,
            'hybrid_processing': (self.sql_engine is not None and 
                                self.rag_engine is not None)
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
            # Fallback classification based on simple patterns
            logger.warning("Query Classifier not available, using fallback classification")
            return self._fallback_classification(query)
        
        try:
            return self.query_classifier.classify_query(query, self.confidence_threshold)
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """Enhanced fallback classification when main classifier fails"""
        query_lower = query.lower()
        
        # Enhanced patterns for fallback
        sql_keywords = [
            'count', 'how many', 'average', 'sum', 'total', 'top', 'most', 'least', 
            'when do', 'what time', 'groups', 'riders', 'passengers', 'trips',
            'moody center', 'downtown', 'saturday', 'age', 'year-old'
        ]
        rag_keywords = [
            'tell me about', 'describe', 'who is', 'what is', 'user', 'profile',
            'details about', 'information about', 'explain'
        ]
        hybrid_keywords = [
            'and tell', 'and describe', 'show me', 'profiles', 'and their',
            'users by', 'top users', 'behavior'
        ]
        
        # Check for hybrid patterns first (most specific)
        if any(keyword in query_lower for keyword in hybrid_keywords):
            classification = QueryType.HYBRID
            confidence = 0.7
        elif any(keyword in query_lower for keyword in sql_keywords):
            classification = QueryType.SQL
            confidence = 0.6
        elif any(keyword in query_lower for keyword in rag_keywords):
            classification = QueryType.RAG
            confidence = 0.6
        else:
            # Default to SQL for numerical/analytical queries
            classification = QueryType.SQL
            confidence = 0.5
        
        return {
            'classification': classification,
            'confidence': confidence,
            'method': 'fallback_classification',
            'error': 'Main classifier unavailable'
        }
    
    def process_sql_query(self, query: str) -> Dict[str, Any]:
        """
        Process query using SQL Engine with enhanced validation
        
        Args:
            query: User's natural language query
            
        Returns:
            SQL processing results with validation
        """
        if not self.sql_engine:
            return {
                'success': False,
                'error': 'SQL Engine not initialized',
                'data': None,
                'query_type': 'SQL',
                'suggestion': 'Check database connection and OpenAI API key'
            }
        
        try:
            result = self.sql_engine.process_natural_language_query(query)
            result['query_type'] = 'SQL'
            
            # Enhanced data validation
            if result.get('success') and result.get('data') is not None:
                result = self._validate_and_enhance_sql_results(result, query)
            
            return result
        except Exception as e:
            logger.error(f"Error in SQL processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None,
                'query_type': 'SQL',
                'natural_query': query,
                'suggestion': 'Try rephrasing your question or check for data availability'
            }
    
    def _validate_and_enhance_sql_results(self, sql_result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Enhanced validation and enhancement of SQL results"""
        try:
            data = sql_result.get('data')
            
            # Basic validation
            if hasattr(data, 'empty') and data.empty:
                sql_result['validation_notes'] = 'Query returned no results'
                sql_result['suggestion'] = self._suggest_alternative_query(query)
            elif hasattr(data, '__len__') and len(data) == 0:
                sql_result['validation_notes'] = 'Query returned empty dataset'
                sql_result['suggestion'] = 'Try broadening your search criteria'
            else:
                sql_result['validation_notes'] = 'Results validated successfully'
            
            # Data quality checks
            if hasattr(data, 'isnull'):
                null_counts = data.isnull().sum().sum()
                if null_counts > 0:
                    sql_result['data_quality_warning'] = f'Dataset contains {null_counts} null values'
                
                # Specific checks for common issues
                if 'Age' in data.columns:
                    null_ages = data['Age'].isnull().sum()
                    if null_ages > 0:
                        sql_result['age_data_warning'] = f'{null_ages} records have missing age data'
            
            # Reasonableness checks
            if hasattr(data, 'select_dtypes'):
                numeric_data = data.select_dtypes(include=['number'])
                for col in numeric_data.columns:
                    if 'passenger' in col.lower() or 'group' in col.lower():
                        max_val = numeric_data[col].max()
                        if max_val > 50:  # Unusually large group
                            sql_result['data_anomaly'] = f'Unusually large values found in {col}: max={max_val}'
            
            return sql_result
            
        except Exception as e:
            logger.warning(f"Error validating SQL results: {e}")
            sql_result['validation_notes'] = f'Validation failed: {str(e)}'
            return sql_result
    
    def _suggest_alternative_query(self, query: str) -> str:
        """Suggest alternative queries when results are empty"""
        query_lower = query.lower()
        
        if 'last month' in query_lower:
            return 'Try asking about "recent trips" or "this year" instead of "last month"'
        elif 'moody center' in query_lower:
            return 'Try searching for "moody" or check other Austin venues'
        elif 'downtown' in query_lower:
            return 'Try searching for "6th street" or "5th street" instead'
        elif 'age' in query_lower or 'year' in query_lower:
            return 'Note: Some users have missing age data. Try broader age ranges'
        else:
            return 'Try using broader search terms or different time periods'
    
    def process_rag_query(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Process query using RAG Engine with enhanced error handling
        
        Args:
            query: User's natural language query
            top_k: Number of results to retrieve
            
        Returns:
            RAG processing results
        """
        if not self.rag_engine:
            return {
                'success': False,
                'error': 'RAG Engine not initialized - embeddings may not be built',
                'context_snippets': [],
                'query_type': 'RAG',
                'suggestion': 'Try running: python embeddings/build_embeddings.py'
            }
        
        if top_k is None:
            top_k = self.default_top_k
        
        try:
            result = self.rag_engine.retrieve_context(query, top_k=top_k)
            
            # Format for consistent interface
            formatted_result = {
                'success': result['returned_results'] > 0,
                'query': result['query'],
                'context_snippets': result['context_snippets'],
                'total_results': result['total_results'],
                'returned_results': result['returned_results'],
                'search_metadata': result.get('search_metadata', {}),
                'query_type': 'RAG',
                'error': result.get('error')
            }
            
            # Add quality assessment
            if formatted_result['success']:
                formatted_result['quality_assessment'] = self._assess_rag_quality(result)
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error in RAG processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'context_snippets': [],
                'query_type': 'RAG',
                'query': query,
                'suggestion': 'Check if embeddings are built: python embeddings/build_embeddings.py'
            }
    
    def _assess_rag_quality(self, rag_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of RAG results"""
        try:
            snippets = rag_result.get('context_snippets', [])
            
            if not snippets:
                return {'quality': 'poor', 'reason': 'No context found'}
            
            # Check relevance scores
            scores = [snippet.get('relevance_score', 0) for snippet in snippets]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            
            if max_score > 0.8:
                quality = 'excellent'
            elif max_score > 0.6:
                quality = 'good'
            elif max_score > 0.4:
                quality = 'fair'
            else:
                quality = 'poor'
            
            return {
                'quality': quality,
                'average_relevance': avg_score,
                'max_relevance': max_score,
                'snippet_count': len(snippets)
            }
            
        except Exception as e:
            logger.warning(f"Error assessing RAG quality: {e}")
            return {'quality': 'unknown', 'error': str(e)}
    
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
            'error': None,
            'processing_summary': {}
        }
        
        try:
            # Step 1: Process with SQL Engine
            logger.info("Step 1: Processing with SQL Engine")
            sql_result = self.process_sql_query(query)
            hybrid_result['sql_result'] = sql_result
            
            # Step 2: Process with RAG Engine (if available)
            logger.info("Step 2: Processing with RAG Engine")
            rag_result = self.process_rag_query(query, top_k)
            hybrid_result['rag_result'] = rag_result
            
            # Step 3: Enhanced entity extraction and cross-referencing
            logger.info("Step 3: Cross-referencing SQL and RAG results")
            if sql_result['success'] and hasattr(sql_result.get('data'), 'iterrows'):
                enhanced_rag_queries = self._extract_entities_for_rag_enhancement(sql_result['data'])
                
                # Perform additional RAG searches based on SQL results
                if self.rag_engine and enhanced_rag_queries:
                    enhanced_contexts = []
                    for enhanced_query in enhanced_rag_queries[:3]:  # Limit to top 3
                        enhanced_rag = self.process_rag_query(enhanced_query, top_k=3)
                        if enhanced_rag['success']:
                            enhanced_contexts.extend(enhanced_rag['context_snippets'])
                    
                    # Add enhanced contexts to main RAG result
                    if enhanced_contexts:
                        hybrid_result['rag_result']['context_snippets'].extend(enhanced_contexts)
                        hybrid_result['rag_result']['returned_results'] += len(enhanced_contexts)
                        hybrid_result['rag_result']['enhanced_search'] = True
            
            # Step 4: Generate combined insights
            combined_insights = self._generate_enhanced_combined_insights(
                sql_result, 
                hybrid_result['rag_result']
            )
            hybrid_result['combined_insights'] = combined_insights
            
            # Step 5: Create processing summary
            hybrid_result['processing_summary'] = self._create_processing_summary(
                sql_result, hybrid_result['rag_result']
            )
            
            # Determine overall success
            hybrid_result['success'] = (
                sql_result.get('success', False) or 
                hybrid_result['rag_result'].get('success', False)
            )
            
            if not hybrid_result['success']:
                errors = []
                if not sql_result.get('success'):
                    errors.append(f"SQL: {sql_result.get('error', 'Unknown error')}")
                if not hybrid_result['rag_result'].get('success'):
                    errors.append(f"RAG: {hybrid_result['rag_result'].get('error', 'Unknown error')}")
                hybrid_result['error'] = '; '.join(errors)
            
            logger.info(f"Hybrid processing completed. Success: {hybrid_result['success']}")
            return hybrid_result
            
        except Exception as e:
            logger.error(f"Error in hybrid processing: {e}")
            hybrid_result['error'] = str(e)
            return hybrid_result
    
    def _extract_entities_for_rag_enhancement(self, sql_data) -> List[str]:
        """
        Enhanced entity extraction from SQL results for RAG search
        
        Args:
            sql_data: DataFrame or dict from SQL results
            
        Returns:
            List of enhanced query strings for RAG
        """
        enhanced_queries = []
        
        try:
            if hasattr(sql_data, 'iterrows'):  # DataFrame
                # Extract specific user IDs for detailed RAG search
                if 'UserID' in sql_data.columns:
                    for user_id in sql_data['UserID'].head(3):  # Top 3 users
                        enhanced_queries.append(f"user {user_id} profile details")
                
                if 'TripID' in sql_data.columns:
                    for trip_id in sql_data['TripID'].head(3):  # Top 3 trips
                        enhanced_queries.append(f"trip {trip_id} details")
                
                # Extract location-based queries with better filtering
                if 'PickUpAddress' in sql_data.columns:
                    for address in sql_data['PickUpAddress'].dropna().head(2):
                        if address and not address.startswith('#') and len(address) > 5:
                            # Extract key location terms
                            location_terms = self._extract_location_terms(address)
                            if location_terms:
                                enhanced_queries.append(f"trips from {location_terms}")
                
                if 'DropOffAddress' in sql_data.columns:
                    for address in sql_data['DropOffAddress'].dropna().head(2):
                        if address and not address.startswith('#') and len(address) > 5:
                            location_terms = self._extract_location_terms(address)
                            if location_terms:
                                enhanced_queries.append(f"trips to {location_terms}")
                
        except Exception as e:
            logger.warning(f"Error extracting entities for RAG enhancement: {e}")
        
        return enhanced_queries
    
    def _extract_location_terms(self, address: str) -> str:
        """Extract key location terms from full address"""
        try:
            # Common Austin landmarks and areas
            landmarks = ['moody', 'downtown', '6th', '5th', 'campus', 'university', 'airport']
            
            address_lower = address.lower()
            for landmark in landmarks:
                if landmark in address_lower:
                    return landmark
            
            # Extract first meaningful part of address
            parts = address.split(',')
            if parts:
                first_part = parts[0].strip()
                if len(first_part) > 3 and not first_part.isdigit():
                    return first_part[:30]  # Limit length
            
            return ""
        except:
            return ""
    
    def _generate_enhanced_combined_insights(self, sql_result: Dict, rag_result: Dict) -> List[Dict[str, Any]]:
        """
        Generate enhanced combined insights from SQL and RAG results
        
        Args:
            sql_result: SQL processing results
            rag_result: RAG processing results
            
        Returns:
            List of combined insights
        """
        insights = []
        
        try:
            # Data availability insight
            sql_success = sql_result.get('success', False)
            rag_success = rag_result.get('success', False)
            
            if sql_success and rag_success:
                insights.append({
                    'type': 'data_availability',
                    'title': 'Comprehensive Analysis Available',
                    'content': f'Both structured data ({sql_result.get("row_count", 0)} records) and contextual information ({rag_result.get("returned_results", 0)} snippets) found',
                    'confidence': 0.9
                })
            elif sql_success:
                insights.append({
                    'type': 'data_availability',
                    'title': 'Structured Data Analysis',
                    'content': f'Found {sql_result.get("row_count", 0)} records in structured data, but limited contextual information available',
                    'confidence': 0.7
                })
            elif rag_success:
                insights.append({
                    'type': 'data_availability',
                    'title': 'Contextual Information Available',
                    'content': f'Found {rag_result.get("returned_results", 0)} relevant context snippets, but no structured data matches',
                    'confidence': 0.6
                })
            
            # Data quality insight
            if sql_success and sql_result.get('data_quality_warning'):
                insights.append({
                    'type': 'data_quality',
                    'title': 'Data Quality Notice',
                    'content': sql_result['data_quality_warning'],
                    'confidence': 0.8
                })
            
            # Cross-reference insight
            if sql_success and rag_success:
                sql_data = sql_result.get('data')
                if hasattr(sql_data, 'columns') and 'UserID' in sql_data.columns:
                    unique_users = len(sql_data['UserID'].unique())
                    insights.append({
                        'type': 'cross_reference',
                        'title': 'Entity Cross-Reference',
                        'content': f'Analysis covers {unique_users} unique users with both quantitative metrics and qualitative context',
                        'confidence': 0.8
                    })
            
        except Exception as e:
            logger.warning(f"Error generating combined insights: {e}")
            insights.append({
                'type': 'error',
                'title': 'Insight Generation Error',
                'content': f'Unable to generate insights: {str(e)}',
                'confidence': 0.1
            })
        
        return insights
    
    def _create_processing_summary(self, sql_result: Dict, rag_result: Dict) -> Dict[str, Any]:
        """Create a summary of the processing results"""
        return {
            'sql_processing': {
                'success': sql_result.get('success', False),
                'records_found': sql_result.get('row_count', 0),
                'execution_time': sql_result.get('execution_time', 0),
                'data_quality': sql_result.get('validation_notes', 'Unknown')
            },
            'rag_processing': {
                'success': rag_result.get('success', False),
                'snippets_found': rag_result.get('returned_results', 0),
                'quality_assessment': rag_result.get('quality_assessment', {}),
                'enhanced_search': rag_result.get('enhanced_search', False)
            },
            'overall_quality': self._assess_overall_quality(sql_result, rag_result)
        }
    
    def _assess_overall_quality(self, sql_result: Dict, rag_result: Dict) -> str:
        """Assess the overall quality of the hybrid processing"""
        sql_success = sql_result.get('success', False)
        rag_success = rag_result.get('success', False)
        
        if sql_success and rag_success:
            sql_rows = sql_result.get('row_count', 0)
            rag_snippets = rag_result.get('returned_results', 0)
            
            if sql_rows > 0 and rag_snippets > 0:
                return 'excellent'
            elif sql_rows > 0 or rag_snippets > 0:
                return 'good'
            else:
                return 'poor'
        elif sql_success or rag_success:
            return 'fair'
        else:
            return 'failed'
    
    def process_query(self, query: str, force_type: str = None, **kwargs) -> Dict[str, Any]:
        """
        Main query processing pipeline with enhanced error handling
        
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
            'results': {},
            'system_status': self._assess_core_functionality()
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
                    # Fallback to SQL if classification fails
                    logger.warning("Classification failed, defaulting to SQL")
                    result['classification'] = {
                        'classification': QueryType.SQL,
                        'confidence': 0.5,
                        'method': 'fallback_default'
                    }
            
            # Step 2: Route to appropriate engine(s)
            query_type = (result['classification']['classification'].value 
                         if result['classification']['classification'] else 'SQL')
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
    
    def generate_response(self, query: str, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate natural language response from processing results
        
        Args:
            query: Original user query
            processing_result: Results from process_query
            
        Returns:
            Generated response
        """
        if not self.response_generator:
            # Fallback response generation
            return {
                'success': False,
                'error': 'Response generator not available',
                'response': self._generate_simple_fallback_response(query, processing_result)
            }
        
        try:
            return self.response_generator.generate_response(query, processing_result)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': self.response_generator.generate_fallback_response(query, processing_result)
            }
    
    def _generate_simple_fallback_response(self, query: str, processing_result: Dict[str, Any]) -> str:
        """Generate a simple fallback response when response generator is unavailable"""
        try:
            if not processing_result.get('success'):
                return f"I apologize, but I couldn't process your query about Fetii's rideshare data. Error: {processing_result.get('error', 'Unknown error')}"
            
            processing_type = processing_result.get('processing_type')
            results = processing_result.get('results', {})
            
            if processing_type == 'SQL':
                row_count = results.get('row_count', 0)
                if row_count > 0:
                    return f"I found {row_count} records matching your query about Fetii's rideshare data. The SQL analysis completed successfully."
                else:
                    return "Your query was processed successfully, but no matching records were found in the Fetii database."
            
            elif processing_type == 'RAG':
                snippet_count = results.get('returned_results', 0)
                if snippet_count > 0:
                    return f"I found {snippet_count} relevant pieces of information about your query in the Fetii knowledge base."
                else:
                    return "I searched the Fetii knowledge base but couldn't find specific information matching your query."
            
            elif processing_type == 'HYBRID':
                sql_success = results.get('sql_result', {}).get('success', False)
                rag_success = results.get('rag_result', {}).get('success', False)
                
                if sql_success and rag_success:
                    return "I was able to analyze both the structured Fetii data and find relevant contextual information for your query."
                elif sql_success:
                    return "I found relevant data in the Fetii database for your query."
                elif rag_success:
                    return "I found relevant contextual information for your query in the Fetii knowledge base."
                else:
                    return "I processed your query but couldn't find specific matching information in the Fetii system."
            
            return "Your query was processed successfully."
            
        except Exception as e:
            logger.error(f"Error in simple fallback response: {e}")
            return "I apologize, but I'm experiencing technical difficulties processing your query about Fetii's rideshare data."

def main():
    """Test the Hybrid Controller functionality"""
    print("Testing Enhanced Hybrid Controller")
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
    print("  Core Functionality:")
    for func, available in status['core_functionality'].items():
        print(f"    {func}: {'✓' if available else '✗'}")
    
    print(f"\n{'='*60}")
    print("Testing Query Processing Pipeline")
    print("-" * 60)
    
    # Test queries for different types
    test_queries = [
        ("How many groups went to Moody Center last month?", None),
        ("What are the top drop-off spots for 18-24 year-olds on Saturday nights?", None),
        ("When do large groups (6+ riders) typically ride downtown?", None),
        ("Tell me about user 168928", "RAG"),
        ("Show me the top 5 users by trip count and their profiles", "HYBRID"),
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
                    if sql_results.get('validation_notes'):
                        print(f"   Validation: {sql_results['validation_notes']}")
            
            elif result['processing_type'] == 'RAG':
                rag_results = result['results']
                print(f"   RAG Results: {rag_results.get('returned_results', 0)} snippets")
                quality = rag_results.get('quality_assessment', {})
                if quality:
                    print(f"   Quality: {quality.get('quality', 'unknown')}")
            
            elif result['processing_type'] == 'HYBRID':
                hybrid_results = result['results']
                sql_success = hybrid_results.get('sql_result', {}).get('success', False)
                rag_success = hybrid_results.get('rag_result', {}).get('success', False)
                print(f"   Hybrid Results: SQL={sql_success}, RAG={rag_success}")
                print(f"   Combined Insights: {len(hybrid_results.get('combined_insights', []))}")
                summary = hybrid_results.get('processing_summary', {})
                if summary:
                    print(f"   Overall Quality: {summary.get('overall_quality', 'unknown')}")
        else:
            print(f"   Error: {result['error']}")
    
    print(f"\n{'='*60}")
    print("Enhanced Hybrid Controller Testing Complete")

if __name__ == "__main__":
    main()
