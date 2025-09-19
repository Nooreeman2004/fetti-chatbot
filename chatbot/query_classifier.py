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
    OFF_TOPIC = "OFF_TOPIC"

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
        self.off_topic_patterns = self._initialize_off_topic_patterns()
        
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
            # Fetii-specific patterns
            {
                'pattern': r'\b(how\s+many\s+(groups?|trips?|users?|passengers?|rides?)|number\s+of\s+(groups?|trips?|users?|passengers?|rides?))\b',
                'flags': re.IGNORECASE,
                'weight': 0.9,
                'description': 'Fetii counting queries'
            },
            # Location-based queries
            {
                'pattern': r'\b(top\s+(drop.?off|pickup|destination|location)|most\s+popular\s+(spot|location|destination)|where\s+do\s+(people|groups|users)\s+go)\b',
                'flags': re.IGNORECASE,
                'weight': 0.9,
                'description': 'Location analysis queries'
            },
            # Time-based patterns
            {
                'pattern': r'\b(when\s+do\s+(people|groups|users|large\s+groups?)\s+(ride|travel|go)|peak\s+(time|hour|day)|busiest\s+(time|hour|day))\b',
                'flags': re.IGNORECASE,
                'weight': 0.9,
                'description': 'Time-based analysis queries'
            },
            # Demographic analysis
            {
                'pattern': r'\b(age\s+(group|range|bracket)|demographics?|young|old|teenagers?|adults?|seniors?)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Demographic analysis queries'
            },
            # Group size analysis
            {
                'pattern': r'\b(large\s+groups?|group\s+size|passenger\s+count|how\s+many\s+people|crowd|party)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Group size analysis queries'
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
                'pattern': r'\b(daily|weekly|monthly|yearly|per\s+day|per\s+week|per\s+month|over\s+time)\b',
                'flags': re.IGNORECASE,
                'weight': 0.7,
                'description': 'Time-based aggregations'
            },
            # Ranking and ordering
            {
                'pattern': r'\b(top|bottom|highest|lowest|rank|ranking|order\s+by|sort\s+by|best|worst)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Ranking operations'
            },
            # Filtering with conditions
            {
                'pattern': r'\b(where|filter|filtered|condition|criteria|between|greater\s+than|less\s+than|equal\s+to)\b',
                'flags': re.IGNORECASE,
                'weight': 0.6,
                'description': 'Filtering operations'
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
                'weight': 0.8,
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
            }
        ]
    
    def _initialize_hybrid_patterns(self) -> List[Dict[str, any]]:
        """Initialize patterns that indicate hybrid queries"""
        return [
            # Comparative analysis
            {
                'pattern': r'\b(compare.*and.*show|analyze.*with.*details|breakdown.*with.*information)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Comparative analysis'
            },
            # Complex analytical queries
            {
                'pattern': r'\b(analyze.*patterns?|trends?.*with.*details|insights?.*about.*specific)\b',
                'flags': re.IGNORECASE,
                'weight': 0.7,
                'description': 'Complex analytical queries'
            },
            # Questions with multiple parts
            {
                'pattern': r'\b(show.*and.*tell|count.*and.*describe|list.*and.*explain)\b',
                'flags': re.IGNORECASE,
                'weight': 0.8,
                'description': 'Multi-part queries'
            },
            # Contextual analytics
            {
                'pattern': r'\b(summarize.*with.*numbers|overview.*with.*statistics|profile.*with.*metrics)\b',
                'flags': re.IGNORECASE,
                'weight': 0.7,
                'description': 'Contextual analytics'
            }
        ]
    
    def _initialize_off_topic_patterns(self) -> List[Dict[str, any]]:
        """Initialize patterns that indicate off-topic queries"""
        return [
            # Programming/Technology questions
            {
                'pattern': r'\b(what\s+is\s+(python|javascript|fastapi|react|django|flask|api|programming|coding|software|framework|library|database|sql|html|css))\b',
                'flags': re.IGNORECASE,
                'weight': 1.0,
                'description': 'Programming/technology questions'
            },
            # Weather queries
            {
                'pattern': r'\b(weather|temperature|forecast|rain|sunny|cloudy|storm|climate)\b',
                'flags': re.IGNORECASE,
                'weight': 1.0,
                'description': 'Weather-related questions'
            },
            # General knowledge questions
            {
                'pattern': r'\b(what\s+is\s+the\s+(capital|population|president|time|date)|who\s+is\s+the\s+(president|prime\s+minister|ceo)|when\s+was\s+.+\s+(founded|created|built))\b',
                'flags': re.IGNORECASE,
                'weight': 1.0,
                'description': 'General knowledge questions'
            },
            # Math/calculation questions (non-data related)
            {
                'pattern': r'\b(what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+|calculate\s+\d+|solve\s+for\s+x|quadratic\s+equation)\b',
                'flags': re.IGNORECASE,
                'weight': 1.0,
                'description': 'Math calculation questions'
            },
            # News/current events
            {
                'pattern': r'\b(latest\s+news|current\s+events|breaking\s+news|today\'s\s+news|headlines)\b',
                'flags': re.IGNORECASE,
                'weight': 1.0,
                'description': 'News/current events questions'
            },
            # Sports/entertainment
            {
                'pattern': r'\b(football\s+score|basketball|soccer|movie|tv\s+show|celebrity|music|song)\b',
                'flags': re.IGNORECASE,
                'weight': 1.0,
                'description': 'Sports/entertainment questions'
            },
            # Health/medical questions
            {
                'pattern': r'\b(medical\s+advice|symptoms|disease|medicine|doctor|health\s+tips|diagnosis)\b',
                'flags': re.IGNORECASE,
                'weight': 1.0,
                'description': 'Health/medical questions'
            },
            # Generic questions without transportation context
            {
                'pattern': r'\b(how\s+to\s+cook|recipe|restaurant|food|shopping|clothing|travel\s+to\s+[^a-zA-Z]*(paris|london|tokyo))\b',
                'flags': re.IGNORECASE,
                'weight': 0.9,
                'description': 'Generic non-transportation questions'
            }
        ]
    
    def _initialize_classification_prompt(self) -> str:
        """Initialize the LLM classification prompt with few-shot examples"""
        return """You are a query classifier for Fetii's transportation data system. Classify user queries into one of four categories:

1. SQL: Queries requiring database operations like aggregations, counts, statistics, filtering, grouping about Fetii transportation data
2. RAG: Queries seeking descriptive information about specific entities, explanations, or contextual details about Fetii transportation data
3. HYBRID: Queries needing both structured data analysis AND contextual information about Fetii transportation data
4. OFF_TOPIC: Queries that are NOT related to Fetii transportation, rideshare, Austin trips, users, or travel data

Examples:

Query: "How many trips were taken last month?"
Classification: SQL
Reason: Requires counting and time-based filtering

Query: "Tell me about user 12345"
Classification: RAG  
Reason: Seeks descriptive information about a specific user

Query: "What's the average trip duration and tell me about the longest trip?"
Classification: HYBRID
Reason: Needs statistical calculation AND descriptive details

Query: "Show me the top 5 users by trip count and their profiles"
Classification: HYBRID
Reason: Requires ranking/aggregation AND detailed user information

Query: "Count trips by destination"
Classification: SQL
Reason: Aggregation with grouping operation

Query: "Explain the transportation patterns in the dataset"
Classification: RAG
Reason: Seeks explanatory/contextual information

Query: "Find users over 30 and describe their travel behavior"
Classification: HYBRID
Reason: Filtering operation AND descriptive analysis

Query: "What is FastAPI?"
Classification: OFF_TOPIC
Reason: Programming question, not related to transportation data

Query: "What's the weather today?"
Classification: OFF_TOPIC
Reason: Weather question, not related to transportation data

Query: "How to cook pasta?"
Classification: OFF_TOPIC
Reason: Cooking question, not related to transportation data

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
        # First check for off-topic patterns
        off_topic_score, off_topic_matches = self._calculate_pattern_score(query, self.off_topic_patterns)
        
        # If off-topic score is high, classify as OFF_TOPIC immediately
        if off_topic_score >= 0.9:
            return {
                'classification': QueryType.OFF_TOPIC,
                'confidence': min(off_topic_score, 1.0),
                'scores': {QueryType.OFF_TOPIC.value: off_topic_score},
                'matched_patterns': {
                    'OFF_TOPIC': off_topic_matches
                },
                'method': 'rule_based'
            }
        
        # Calculate scores for each type
        sql_score, sql_matches = self._calculate_pattern_score(query, self.sql_patterns)
        rag_score, rag_matches = self._calculate_pattern_score(query, self.rag_patterns)
        hybrid_score, hybrid_matches = self._calculate_pattern_score(query, self.hybrid_patterns)
        
        # Determine classification based on scores
        scores = {
            QueryType.SQL: sql_score,
            QueryType.RAG: rag_score,
            QueryType.HYBRID: hybrid_score,
            QueryType.OFF_TOPIC: off_topic_score
        }
        
        # Find the highest scoring category
        max_score = max(scores.values())
        
        if max_score == 0:
            classification = None
            confidence = 0.0
        else:
            classification = max(scores, key=scores.get)
            confidence = min(max_score / (max_score + 1), 1.0)  # Normalize confidence
        
        return {
            'classification': classification,
            'confidence': confidence,
            'scores': {k.value: v for k, v in scores.items()},
            'matched_patterns': {
                'SQL': sql_matches,
                'RAG': rag_matches,
                'HYBRID': hybrid_matches,
                'OFF_TOPIC': off_topic_matches
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
                    {"role": "system", "content": "You are a precise query classifier. Respond only with 'SQL', 'RAG', or 'HYBRID' followed by a brief reason."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Extract classification
            classification_str = None
            for query_type in ['SQL', 'RAG', 'HYBRID']:
                if query_type in response_text.upper():
                    classification_str = query_type
                    break
            
            if classification_str:
                classification = QueryType(classification_str)
                confidence = 0.8  # High confidence in LLM classification
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
    
    def classify_query(self, query: str, confidence_threshold: float = 0.5) -> Dict[str, any]:
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
            'classifications': {
                'SQL': 0,
                'RAG': 0, 
                'HYBRID': 0,
                'UNCLASSIFIED': 0
            },
            'methods': {
                'rule_based': 0,
                'llm': 0,
                'hybrid_rule_llm': 0,
                'rule_based_fallback': 0,
                'error': 0
            },
            'avg_confidence': 0.0
        }
        
        total_confidence = 0.0
        
        for result in results:
            # Count classifications
            if result.get('classification'):
                classification_name = result['classification'].value
                stats['classifications'][classification_name] += 1
            else:
                stats['classifications']['UNCLASSIFIED'] += 1
            
            # Count methods
            method = result.get('method', 'error')
            if method in stats['methods']:
                stats['methods'][method] += 1
            else:
                stats['methods']['error'] += 1
            
            # Sum confidence
            total_confidence += result.get('confidence', 0.0)
        
        # Calculate average confidence
        if results:
            stats['avg_confidence'] = total_confidence / len(results)
        
        return stats

def main():
    """Test the query classifier with sample queries"""
    
    # Test queries
    test_queries = [
        "How many trips were taken last month?",
        "Tell me about user 12345",
        "What's the average trip duration?",
        "Describe the transportation patterns",
        "Show me the top 5 users by trip count and their profiles",
        "Count trips by destination",
        "Find users over 30 years old",
        "Explain how the booking system works",
        "Compare trip durations between different age groups",
        "What is the total number of passengers across all trips?"
    ]
    
    # Initialize classifier
    classifier = QueryClassifier()
    
    print("Testing Query Classification System")
    print("=" * 50)
    
    # Test individual classifications
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        result = classifier.classify_query(query)
        
        if result.get('classification'):
            print(f"   Classification: {result['classification'].value}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Method: {result['method']}")
            
            if 'matched_patterns' in result.get('rule_based_result', {}):
                patterns = result['rule_based_result']['matched_patterns']
                for pattern_type, matches in patterns.items():
                    if matches:
                        print(f"   {pattern_type} patterns: {', '.join(matches)}")
        else:
            print(f"   Classification: UNCLASSIFIED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Batch classification and stats
    print(f"\n{'='*50}")
    print("Batch Classification Statistics")
    print("=" * 50)
    
    batch_results = classifier.batch_classify(test_queries)
    stats = classifier.get_classification_stats(batch_results)
    
    print(f"Total queries: {stats['total_queries']}")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")
    print("\nClassification distribution:")
    for classification, count in stats['classifications'].items():
        print(f"  {classification}: {count}")
    
    print("\nMethod distribution:")
    for method, count in stats['methods'].items():
        print(f"  {method}: {count}")

if __name__ == "__main__":
    main()