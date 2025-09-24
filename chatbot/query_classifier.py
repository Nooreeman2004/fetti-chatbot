import re
import openai
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List, Tuple, Optional
import json
from enum import Enum

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Enum for query classification types"""
    SQL = "SQL"
    RAG = "RAG" 
    HYBRID = "HYBRID"

class QueryClassifier:
    """
    Query Classification Layer for determining processing method
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the query classifier
        
        Args:
            openai_api_key: OpenAI API key (if None, will try to load from environment)
        """
        # Load OpenAI API key
        if openai_api_key:
            self.openai_api_key = openai_api_key
        else:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found. LLM fallback will not be available.")
        else:
            openai.api_key = self.openai_api_key
        
        # Define rule-based patterns
        self.sql_patterns = self._initialize_sql_patterns()
        self.rag_patterns = self._initialize_rag_patterns()
        self.hybrid_patterns = self._initialize_hybrid_patterns()
        
        # LLM classification prompt
        self.classification_prompt = self._initialize_classification_prompt()
    
    def _initialize_sql_patterns(self) -> List[Dict[str, any]]:
        """Initialize patterns that indicate SQL queries"""
        return [
            # Aggregation functions
            {
                'pattern': r'\b(average|avg|mean|sum|total|count|max|maximum|min|minimum|median)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Aggregation functions'
            },
            # Group by operations
            {
                'pattern': r'\b(group\s+by|grouped?\s+by|categorize|breakdown|segment)\b',
                'flags': re.IGNORECASE,
                'weight': 0.9,
                'description': 'Grouping operations'
            },
            # Statistical operations
            {
                'pattern': r'\b(statistics|stats|analytics|metrics|percentage|percent|ratio|compare|comparison)\b',
                'flags': re.IGNORECASE,
                'weight': 0.7,
                'description': 'Statistical operations'
            },
            # Numerical queries
            {
                'pattern': r'\b(how\s+many|number\s+of|quantity|amount|total\s+number)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Numerical queries'
            },
            # Time-based aggregations
            {
                'pattern': r'\b(daily|weekly|monthly|yearly|per\s+day|per\s+week|per\s+month|over\s+time|last\s+month|this\s+month)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Time-based aggregations'
            },
            # Ranking and ordering
            {
                'pattern': r'\b(top|bottom|highest|lowest|rank|ranking|order\s+by|sort\s+by|best|worst|most|least)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Ranking operations'
            },
            # Filtering with conditions
            {
                'pattern': r'\b(where|filter|filtered|condition|criteria|between|greater\s+than|less\s+than|equal\s+to|over|under|above|below)\b',
                'flags': re.IGNORECASE,
                'weight': 0.6,
                'description': 'Filtering operations'
            },
            # Fetii-specific group analysis
            {
                'pattern': r'\b(groups?|riders?|passengers?|group\s+size|large\s+groups?|small\s+groups?|\d+\+\s+riders?|\d+\+\s+passengers?)\b',
                'flags': re.IGNORECASE,
                'weight': 0.9,
                'description': 'Group size analysis'
            },
            # Austin location-based counting
            {
                'pattern': r'\b(went\s+to|trips?\s+to|rides?\s+to|destinations?|drop.?off\s+spots?|pickup\s+spots?)\b',
                'flags': re.IGNORECASE,
                'weight': 0.7,
                'description': 'Location-based counting'
            },
            # Time pattern analysis
            {
                'pattern': r'\b(when\s+do|what\s+time|saturday\s+nights?|weekends?|peak\s+times?|typically)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Time pattern analysis'
            },
            # Age group analysis
            {
                'pattern': r'\b(\d+.?\d*\s*year.?olds?|\d+.?\d*\s*to\s*\d+.?\d*|age\s+groups?|demographics?)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Age group analysis'
            },
            # Austin-specific locations
            {
                'pattern': r'\b(moody\s+center|downtown|6th\s+street|5th\s+street|campus|university|airport|austin)\b',
                'flags': re.IGNORECASE,
                'weight': 0.7,
                'description': 'Austin location references'
            }
        ]
    
    def _initialize_rag_patterns(self) -> List[Dict[str, any]]:
        """Initialize patterns that indicate RAG queries"""
        return [
            # Descriptive queries
            {
                'pattern': r'\b(tell\s+me\s+about|describe|what\s+is|who\s+is|information\s+about|details\s+about)\b',
                'flags': re.IGNORECASE,
                'weight': 0.9,
                'description': 'Descriptive queries'
            },
            # Specific entity queries
            {
                'pattern': r'\b(user\s+\d+|trip\s+\d+|customer\s+\d+|profile\s+of|background\s+of)\b',
                'flags': re.IGNORECASE,
                'weight': 0.9,
                'description': 'Specific entity queries'
            },
            # Explanatory queries
            {
                'pattern': r'\b(explain|why|how\s+does|what\s+does|meaning\s+of|definition\s+of)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Explanatory queries'
            },
            # Narrative queries
            {
                'pattern': r'\b(story|history|background|journey|experience|narrative)\b',
                'flags': re.IGNORECASE,
                'weight': 0.7,
                'description': 'Narrative queries'
            },
            # Context-seeking queries
            {
                'pattern': r'\b(context|situation|scenario|case|example|instance)\b',
                'flags': re.IGNORECASE,
                'weight': 0.6,
                'description': 'Context-seeking queries'
            },
            # General information queries
            {
                'pattern': r'\b(what\s+are|what\s+kind|overview|summary|general\s+info)\b',
                'flags': re.IGNORECASE,
                'weight': 0.6,
                'description': 'General information queries'
            }
        ]
    
    def _initialize_hybrid_patterns(self) -> List[Dict[str, any]]:
        """Initialize patterns that indicate hybrid queries"""
        return [
            # Comparative analysis with details
            {
                'pattern': r'\b(show.*and.*tell|list.*and.*describe|count.*and.*explain|find.*and.*describe)\b',
                'flags': re.IGNORECASE,
                'weight': 0.9,
                'description': 'Multi-part queries requiring both data and context'
            },
            # Top/ranking with profiles
            {
                'pattern': r'\b(top.*users?.*profiles?|top.*and.*their|show.*top.*and.*tell|users?.*by.*count.*and.*profiles?)\b',
                'flags': re.IGNORECASE,
                'weight': 0.9,
                'description': 'Ranking with profile information'
            },
            # Analysis with behavior description
            {
                'pattern': r'\b(analyze.*patterns?|trends?.*with.*details|insights?.*about.*specific|behavior|travel\s+behavior)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Behavioral analysis queries'
            },
            # Demographic analysis with specifics
            {
                'pattern': r'\b(users?.*over.*and.*describe|age.*groups?.*and|demographics?.*and.*behavior)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Demographic analysis with context'
            },
            # Complex Fetii-specific queries
            {
                'pattern': r'\b(groups?.*and.*profiles?|riders?.*and.*details|passengers?.*and.*information)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Group analysis with detailed information'
            }
        ]
    
    def _initialize_classification_prompt(self) -> str:
        """Initialize the LLM classification prompt with Fetii-specific examples"""
        return """You are a query classifier for Fetii's Austin group rideshare data system. Classify user queries into one of three categories:

1. SQL: Queries requiring database operations like aggregations, counts, statistics, filtering, grouping, rankings
2. RAG: Queries seeking descriptive information about specific entities, explanations, or contextual details  
3. HYBRID: Queries needing both structured data analysis AND contextual information

Fetii-Specific Examples:

Query: "How many groups went to Moody Center last month?"
Classification: SQL
Reason: Requires counting trips with location and time filtering

Query: "What are the top drop-off spots for 18–24 year-olds on Saturday nights?"
Classification: SQL
Reason: Requires aggregation, age filtering, time filtering, and ranking

Query: "When do large groups (6+ riders) typically ride downtown?"
Classification: SQL
Reason: Requires time analysis with group size and location filtering

Query: "Tell me about user 12345"
Classification: RAG  
Reason: Seeks descriptive information about a specific user

Query: "Show me the top 5 users by trip count and their profiles"
Classification: HYBRID
Reason: Requires ranking/aggregation (SQL) AND detailed user information (RAG)

Query: "Find users over 30 and describe their travel behavior"
Classification: HYBRID
Reason: Filtering operation (SQL) AND descriptive behavioral analysis (RAG)

Query: "What's the average trip duration?"
Classification: SQL
Reason: Simple aggregation operation

Query: "Explain the transportation patterns in Austin"
Classification: RAG
Reason: Seeks explanatory/contextual information

Query: "Count trips by destination and tell me about the most popular ones"
Classification: HYBRID
Reason: Aggregation (SQL) AND descriptive details about locations (RAG)

Now classify this query:
Query: "{query}"
Classification:"""
    
    def _calculate_pattern_score(self, query: str, patterns: List[Dict]) -> Tuple[float, List[str]]:
        """
        Calculate score based on pattern matching
        
        Args:
            query: User query string
            patterns: List of pattern dictionaries
            
        Returns:
            Tuple of (total_score, matched_descriptions)
        """
        total_score = 0.0
        matched_descriptions = []
        
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            flags = pattern_info.get('flags', 0)
            weight = pattern_info.get('weight', 1.0)
            description = pattern_info.get('description', 'Pattern match')
            
            if re.search(pattern, query, flags):
                total_score += weight
                matched_descriptions.append(description)
        
        return total_score, matched_descriptions
    
    def _rule_based_classification(self, query: str) -> Dict[str, any]:
        """
        Perform rule-based classification
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with classification results
        """
        # Calculate scores for each type
        sql_score, sql_matches = self._calculate_pattern_score(query, self.sql_patterns)
        rag_score, rag_matches = self._calculate_pattern_score(query, self.rag_patterns)
        hybrid_score, hybrid_matches = self._calculate_pattern_score(query, self.hybrid_patterns)
        
        # Boost hybrid score if both SQL and RAG patterns are present
        if sql_score > 0 and rag_score > 0:
            hybrid_score += 0.5
        
        # Determine classification based on scores
        scores = {
            QueryType.SQL: sql_score,
            QueryType.RAG: rag_score,
            QueryType.HYBRID: hybrid_score
        }
        
        # Find the highest scoring category
        max_score = max(scores.values())
        
        if max_score == 0:
            classification = None
            confidence = 0.0
        else:
            classification = max(scores, key=scores.get)
            # Improved confidence calculation
            total_score = sum(scores.values())
            confidence = min(max_score / (total_score + 1), 0.95)  # Cap at 95%
        
        return {
            'classification': classification,
            'confidence': confidence,
            'scores': {k.value: v for k, v in scores.items()},
            'matched_patterns': {
                'SQL': sql_matches,
                'RAG': rag_matches,
                'HYBRID': hybrid_matches
            },
            'method': 'rule_based'
        }
    
    def _llm_classification(self, query: str) -> Dict[str, any]:
        """
        Use LLM for classification when rule-based approach is uncertain
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with LLM classification results
        """
        if not self.openai_api_key:
            logger.warning("OpenAI API key not available for LLM classification")
            return {
                'classification': None,
                'confidence': 0.0,
                'method': 'llm_unavailable',
                'error': 'OpenAI API key not found'
            }
        
        try:
            # Prepare the prompt
            prompt = self.classification_prompt.format(query=query)
            
            # Call OpenAI API
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise query classifier for Fetii rideshare data. Respond only with 'SQL', 'RAG', or 'HYBRID' followed by a brief reason."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Extract classification
            classification_str = None
            for query_type in ['HYBRID', 'SQL', 'RAG']:  # Check HYBRID first as it's more specific
                if query_type in response_text.upper():
                    classification_str = query_type
                    break
            
            if classification_str:
                classification = QueryType(classification_str)
                confidence = 0.85  # High confidence in LLM classification
            else:
                classification = None
                confidence = 0.0
            
            return {
                'classification': classification,
                'confidence': confidence,
                'method': 'llm',
                'llm_response': response_text,
                'llm_reasoning': response_text
            }
            
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return {
                'classification': None,
                'confidence': 0.0,
                'method': 'llm_error',
                'error': str(e)
            }
    
    def classify_query(self, query: str, confidence_threshold: float = 0.6) -> Dict[str, any]:
        """
        Main method to classify a user query
        
        Args:
            query: User query string
            confidence_threshold: Minimum confidence for rule-based classification
            
        Returns:
            Dictionary with classification results
        """
        # Clean and normalize query
        query = query.strip()
        
        if not query:
            return {
                'classification': None,
                'confidence': 0.0,
                'method': 'empty_query',
                'error': 'Empty query provided'
            }
        
        # Step 1: Try rule-based classification
        rule_result = self._rule_based_classification(query)
        
        # If rule-based classification is confident enough, return it
        if (rule_result['classification'] and 
            rule_result['confidence'] >= confidence_threshold):
            
            logger.info(f"Rule-based classification: {rule_result['classification'].value} "
                       f"(confidence: {rule_result['confidence']:.2f})")
            return rule_result
        
        # Step 2: Fallback to LLM classification
        logger.info("Rule-based classification uncertain, using LLM fallback...")
        llm_result = self._llm_classification(query)
        
        # Combine results
        final_result = {
            'query': query,
            'classification': llm_result.get('classification'),
            'confidence': llm_result.get('confidence', 0.0),
            'method': 'hybrid_rule_llm',
            'rule_based_result': rule_result,
            'llm_result': llm_result
        }
        
        # If LLM also failed, fall back to best rule-based guess
        if not final_result['classification'] and rule_result['classification']:
            final_result['classification'] = rule_result['classification']
            final_result['confidence'] = rule_result['confidence']
            final_result['method'] = 'rule_based_fallback'
            logger.warning("LLM classification failed, using rule-based fallback")
        
        return final_result
    
    def batch_classify(self, queries: List[str]) -> List[Dict[str, any]]:
        """
        Classify multiple queries in batch
        
        Args:
            queries: List of query strings
            
        Returns:
            List of classification results
        """
        results = []
        for i, query in enumerate(queries):
            logger.info(f"Classifying query {i+1}/{len(queries)}: {query[:50]}...")
            result = self.classify_query(query)
            results.append(result)
        
        return results
    
    def get_classification_stats(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Get statistics from batch classification results
        
        Args:
            results: List of classification results
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_queries': len(results),
            'successful_classifications': 0,
            'classification_counts': {
                'SQL': 0,
                'RAG': 0,
                'HYBRID': 0,
                'FAILED': 0
            },
            'method_counts': {
                'rule_based': 0,
                'llm': 0,
                'hybrid_rule_llm': 0,
                'rule_based_fallback': 0,
                'failed': 0
            },
            'average_confidence': 0.0
        }
        
        total_confidence = 0.0
        
        for result in results:
            if result.get('classification'):
                stats['successful_classifications'] += 1
                stats['classification_counts'][result['classification'].value] += 1
                total_confidence += result.get('confidence', 0.0)
            else:
                stats['classification_counts']['FAILED'] += 1
            
            method = result.get('method', 'failed')
            if method in stats['method_counts']:
                stats['method_counts'][method] += 1
            else:
                stats['method_counts']['failed'] += 1
        
        if stats['successful_classifications'] > 0:
            stats['average_confidence'] = total_confidence / stats['successful_classifications']
        
        return stats

def main():
    """Test the Query Classifier functionality"""
    print("Testing Query Classifier")
    print("=" * 60)
    
    # Initialize classifier
    classifier = QueryClassifier()
    
    # Test queries
    test_queries = [
        "How many groups went to Moody Center last month?",
        "What are the top drop-off spots for 18-24 year-olds on Saturday nights?",
        "When do large groups (6+ riders) typically ride downtown?",
        "Tell me about user 12345",
        "Show me the top 5 users by trip count and their profiles",
        "Find users over 30 and describe their travel behavior",
        "What's the average trip duration?",
        "Explain the transportation patterns in Austin",
        "Count trips by destination and tell me about the most popular ones"
    ]
    
    print("Testing individual queries:")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        result = classifier.classify_query(query)
        classification = result['classification'].value if result['classification'] else 'FAILED'
        confidence = result.get('confidence', 0.0)
        method = result.get('method', 'unknown')
        
        print(f"{i}. '{query[:50]}...'")
        print(f"   Classification: {classification} (confidence: {confidence:.2f}, method: {method})")
    
    # Batch classification stats
    print(f"\n{'='*60}")
    print("Batch Classification Statistics:")
    print("-" * 40)
    
    batch_results = classifier.batch_classify(test_queries)
    stats = classifier.get_classification_stats(batch_results)
    
    print(f"Total queries: {stats['total_queries']}")
    print(f"Successful classifications: {stats['successful_classifications']}")
    print(f"Average confidence: {stats['average_confidence']:.2f}")
    print(f"Classification distribution: {stats['classification_counts']}")
    print(f"Method distribution: {stats['method_counts']}")

if __name__ == "__main__":
    main()
