import openai
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import json
from datetime import datetime
import re

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Response Generation Layer for converting SQL/RAG results into natural language
    """
    
    def __init__(self, openai_api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the response generator
        
        Args:
            openai_api_key: OpenAI API key (if None, will try to load from environment)
            model: OpenAI model to use for response generation
        """
        # Load OpenAI API key
        if openai_api_key:
            self.openai_api_key = openai_api_key
        else:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found. Response generation will not be available.")
        else:
            openai.api_key = self.openai_api_key
        
        self.model = model
        
        # Initialize prompt templates
        self.system_prompt = self._initialize_system_prompt()
        self.sql_prompt_template = self._initialize_sql_prompt()
        self.rag_prompt_template = self._initialize_rag_prompt()
        self.hybrid_prompt_template = self._initialize_hybrid_prompt()
    
    def _initialize_system_prompt(self) -> str:
        """Initialize the system prompt for the response generator"""
        return ("You are FetiiGPT, a precise data analyst for Fetii Austin rideshare service. "
                "Provide EXACT, concise answers based ONLY on the actual data provided.\n\n"
                "CRITICAL RULES:\n"
                "1. Answer ONLY with data provided - no assumptions or generalizations\n"
                "2. Be precise with numbers - state exact counts, not estimates\n"
                "3. Keep responses under 100 words unless asking for detailed analysis\n"
                "4. Start with the direct answer, then add brief context if needed\n"
                "5. Use specific data points: \"X trips\", \"Y users\", \"Z% of total\"\n"
                "6. If data does not answer the question exactly, say \"Data shows [what it shows] but does not specify [what is missing]\"\n"
                "7. For time-based queries, work with available date data only\n"
                "8. Reference Austin locations by their exact names from the data\n\n"
                "Response format:\n"
                "- Direct answer first (1-2 sentences)\n"
                "- Supporting data (bullet points if multiple facts)\n"
                "- Brief insight (1 sentence max)\n\n"
                "Example:\n"
                "Q: \"How many groups went to Moody Center?\"\n"
                "A: \"Based on the data, 47 trips went to destinations containing 'Moody'. This represents 2.4% of all trips in the dataset.\"")
    
    def _initialize_sql_prompt(self) -> str:
        """Initialize the prompt template for SQL-based responses"""
        return ("Answer this question with EXACT data only:\n\n"
                "Question: {query}\n\n"
                "SQL Results: {sql_results}\n"
                "Row count: {row_count}\n\n"
                "Conversation History (last 7 messages):\n{conversation_context}\n\n"
                "Provide a precise, data-driven answer under 100 words. Start with the exact number/result.")
    
    def _initialize_rag_prompt(self) -> str:
        """Initialize the prompt template for RAG-based responses"""
        return ("Based on the retrieved context information, provide a comprehensive answer to the user question.\n\n"
                "User Question: {query}\n\n"
                "Retrieved Context Snippets:\n{context_snippets}\n\n"
                "Context Summary:\n"
                "- Total results found: {total_results}\n"
                "- Results returned: {returned_results}\n"
                "- Search metadata: {search_metadata}\n\n"
                "Please provide a natural language response that:\n"
                "1. Directly addresses the user question using the context provided\n"
                "2. Synthesizes information from multiple sources when applicable\n"
                "3. Highlights the most relevant details\n"
                "4. Maintains context about the transportation domain\n"
                "5. If multiple entities are mentioned, organize information clearly\n\n"
                "Response:")
    
    def _initialize_hybrid_prompt(self) -> str:
        """Initialize the prompt template for hybrid responses"""
        return ("You have both structured SQL results and contextual information from semantic search. "
                "Provide a comprehensive response that combines both data sources.\n\n"
                "User Question: {query}\n\n"
                "SQL Analysis Results:\n{sql_summary}\n\n"
                "Contextual Information:\n{rag_summary}\n\n"
                "Combined Insights:\n{combined_insights}\n\n"
                "Cross-References:\n{cross_references}\n\n"
                "Please provide a natural language response that:\n"
                "1. Integrates both quantitative (SQL) and qualitative (RAG) insights\n"
                "2. Shows how the structured data and contextual information complement each other\n"
                "3. Provides a complete picture addressing the user question\n"
                "4. Highlights any interesting patterns or correlations found\n"
                "5. Organizes the response logically from general findings to specific details\n\n"
                "Response:")
    
    def format_sql_results(self, sql_result: Dict[str, Any]) -> str:
        """Format SQL results for inclusion in prompts"""
        try:
            if not sql_result.get('success') or sql_result.get('data') is None:
                return f"SQL query failed: {sql_result.get('error', 'Unknown error')}"
            
            data = sql_result['data']
            
            # Handle DataFrame results
            if isinstance(data, pd.DataFrame):
                if len(data) == 0:
                    return "No results found."
                
                # For small results, show all data
                if len(data) <= 10:
                    return data.to_string(index=False)
                
                # For larger results, show summary + sample
                summary = f"Dataset contains {len(data)} rows with columns: {', '.join(data.columns)}\n\n"
                
                # Show first few rows
                sample = data.head(5).to_string(index=False)
                summary += f"First 5 rows:\n{sample}\n\n"
                
                # Add basic statistics for numeric columns
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    stats = data[numeric_cols].describe()
                    summary += f"Summary statistics:\n{stats.to_string()}"
                
                return summary
            
            # Handle dictionary results
            elif isinstance(data, dict):
                return json.dumps(data, indent=2)
            
            # Handle other data types
            else:
                return str(data)
                
        except Exception as e:
            logger.error(f"Error formatting SQL results: {e}")
            return f"Error formatting results: {str(e)}"
    
    def format_rag_context(self, rag_result: Dict[str, Any]) -> str:
        """Format RAG context snippets for inclusion in prompts"""
        try:
            if not rag_result.get('success') or not rag_result.get('context_snippets'):
                return f"No relevant context found: {rag_result.get('error', 'No results')}"
            
            context_text = ""
            snippets = rag_result['context_snippets']
            
            for i, snippet in enumerate(snippets, 1):
                relevance = snippet.get('relevance_score', 0)
                content = snippet.get('content', '')
                snippet_type = snippet.get('type', 'unknown')
                identifiers = snippet.get('identifiers', '')
                additional_info = snippet.get('additional_info', '')
                
                context_text += f"\n--- Context {i} (Relevance: {relevance:.3f}, Type: {snippet_type}) ---\n"
                context_text += f"{content}\n"
                
                if identifiers:
                    context_text += f"Identifiers: {identifiers}\n"
                
                if additional_info:
                    context_text += f"Additional info: {additional_info}\n"
            
            return context_text.strip()
            
        except Exception as e:
            logger.error(f"Error formatting RAG context: {e}")
            return f"Error formatting context: {str(e)}"
    
    def format_hybrid_summary(self, hybrid_result: Dict[str, Any]) -> Dict[str, str]:
        """Format hybrid results into structured summaries"""
        try:
            sql_summary = "No SQL results available."
            rag_summary = "No contextual information available."
            combined_insights = "No combined insights generated."
            cross_references = "No cross-references found."
            
            # Format SQL summary
            if hybrid_result.get('sql_result') and hybrid_result['sql_result'].get('success'):
                sql_data = hybrid_result['sql_result']['data']
                if isinstance(sql_data, pd.DataFrame) and len(sql_data) > 0:
                    sql_summary = f"SQL Analysis found {len(sql_data)} records with {len(sql_data.columns)} attributes. "
                    
                    # Add key statistics
                    numeric_cols = sql_data.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        for col in numeric_cols[:3]:  # Top 3 numeric columns
                            if col in sql_data.columns:
                                mean_val = sql_data[col].mean()
                                sql_summary += f"Average {col}: {mean_val:.2f}. "
            
            # Format RAG summary
            if hybrid_result.get('rag_result') and hybrid_result['rag_result'].get('success'):
                rag_data = hybrid_result['rag_result']
                snippet_count = rag_data.get('returned_results', 0)
                if snippet_count > 0:
                    top_score = max(snippet['relevance_score'] for snippet in rag_data['context_snippets'])
                    rag_summary = f"Retrieved {snippet_count} relevant context snippets with top relevance score of {top_score:.3f}. "
                    
                    # Add context types
                    types = [snippet.get('type', 'unknown') for snippet in rag_data['context_snippets']]
                    type_counts = {t: types.count(t) for t in set(types)}
                    rag_summary += f"Context types: {', '.join([f'{k}({v})' for k, v in type_counts.items()])}."
            
            # Format combined insights
            if hybrid_result.get('combined_insights'):
                insights = hybrid_result['combined_insights']
                insight_summaries = []
                
                for insight in insights:
                    insight_type = insight.get('type', 'unknown')
                    title = insight.get('title', 'Insight')
                    content = insight.get('content', '')
                    insight_summaries.append(f"{title}: {content}")
                
                combined_insights = " | ".join(insight_summaries)
            
            # Cross-references
            cross_ref_info = []
            if hybrid_result.get('sql_result', {}).get('success') and hybrid_result.get('rag_result', {}).get('success'):
                cross_ref_info.append("Both SQL and RAG results available for comprehensive analysis")
                
                # Check for entity overlaps (simplified)
                sql_data = hybrid_result['sql_result'].get('data')
                if isinstance(sql_data, pd.DataFrame) and 'UserID' in sql_data.columns:
                    unique_users = len(sql_data['UserID'].unique())
                    cross_ref_info.append(f"SQL data contains {unique_users} unique users")
            
            cross_references = ". ".join(cross_ref_info) if cross_ref_info else "No specific cross-references identified."
            
            return {
                'sql_summary': sql_summary,
                'rag_summary': rag_summary,
                'combined_insights': combined_insights,
                'cross_references': cross_references
            }
            
        except Exception as e:
            logger.error(f"Error formatting hybrid summary: {e}")
            return {
                'sql_summary': f"Error processing SQL results: {str(e)}",
                'rag_summary': f"Error processing RAG results: {str(e)}",
                'combined_insights': f"Error generating insights: {str(e)}",
                'cross_references': f"Error finding cross-references: {str(e)}"
            }
    
    def format_conversation_context(self, chat_history: List[Dict] = None) -> str:
        """Format the last 7 messages for conversation context"""
        if not chat_history:
            return "No previous conversation."
        
        # Get last 7 messages
        recent_history = chat_history[-7:] if len(chat_history) > 7 else chat_history
        
        context_text = ""
        for i, entry in enumerate(recent_history):
            user_query = entry.get('query', 'Unknown query')
            # Get the response from either processing_result or response_result
            response = None
            if entry.get('response_result', {}).get('success'):
                response = entry['response_result']['response']
            elif entry.get('processing_result', {}).get('success'):
                # Try to extract a short summary from processing result
                processing = entry['processing_result']
                if processing.get('results', {}).get('data') is not None:
                    data = processing['results']['data']
                    if isinstance(data, pd.DataFrame):
                        response = f"Found {len(data)} records"
                    else:
                        response = "Data retrieved"
                else:
                    response = "Query processed"
            else:
                response = "No response available"
            
            context_text += f"Q{i+1}: {user_query}\nA{i+1}: {response[:100]}...\n\n"
        
        return context_text.strip()

    def generate_sql_response(self, query: str, sql_result: Dict[str, Any], chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate response for SQL-only results"""
        if not self.openai_api_key:
            return {
                'success': False,
                'error': 'OpenAI API key not available',
                'response': None
            }
        
        try:
            # Format SQL results
            formatted_results = self.format_sql_results(sql_result)
            
            # Format conversation context
            conversation_context = self.format_conversation_context(chat_history)
            
            # Prepare prompt
            prompt = self.sql_prompt_template.format(
                query=query,
                sql_results=formatted_results,
                row_count=sql_result.get('row_count', 0),
                conversation_context=conversation_context
            )
            
            # Generate response
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            return {
                'success': True,
                'response': generated_response,
                'response_type': 'sql',
                'tokens_used': response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    def generate_rag_response(self, query: str, rag_result: Dict[str, Any], chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate response for RAG-only results"""
        if not self.openai_api_key:
            return {
                'success': False,
                'error': 'OpenAI API key not available',
                'response': None
            }
        
        try:
            # Format RAG context
            formatted_context = self.format_rag_context(rag_result)
            
            # Prepare prompt
            prompt = self.rag_prompt_template.format(
                query=query,
                context_snippets=formatted_context,
                total_results=rag_result.get('total_results', 0),
                returned_results=rag_result.get('returned_results', 0),
                search_metadata=json.dumps(rag_result.get('search_metadata', {}), indent=2)
            )
            
            # Generate response
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            return {
                'success': True,
                'response': generated_response,
                'response_type': 'rag',
                'tokens_used': response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    def generate_hybrid_response(self, query: str, hybrid_result: Dict[str, Any], chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate response for hybrid SQL+RAG results"""
        if not self.openai_api_key:
            return {
                'success': False,
                'error': 'OpenAI API key not available',
                'response': None
            }
        
        try:
            # Format hybrid summaries
            summaries = self.format_hybrid_summary(hybrid_result)
            
            # Prepare prompt
            prompt = self.hybrid_prompt_template.format(
                query=query,
                sql_summary=summaries['sql_summary'],
                rag_summary=summaries['rag_summary'],
                combined_insights=summaries['combined_insights'],
                cross_references=summaries['cross_references']
            )
            
            # Generate response
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            return {
                'success': True,
                'response': generated_response,
                'response_type': 'hybrid',
                'tokens_used': response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating hybrid response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    def generate_response(self, query: str, processing_result: Dict[str, Any], chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Main method to generate natural language response based on processing results
        
        Args:
            query: Original user query
            processing_result: Results from hybrid controller
            
        Returns:
            Dictionary with generated response and metadata
        """
        start_time = datetime.now()
        
        try:
            # Determine processing type and route to appropriate generator
            processing_type = processing_result.get('processing_type', 'unknown')
            results = processing_result.get('results', {})
            
            if processing_type == 'SQL':
                response_result = self.generate_sql_response(query, results, chat_history)
                
            elif processing_type == 'RAG':
                response_result = self.generate_rag_response(query, results, chat_history)
                
            elif processing_type == 'HYBRID':
                response_result = self.generate_hybrid_response(query, results, chat_history)
                
            elif processing_type == 'OFF_TOPIC':
                response_result = self.generate_off_topic_response(query)
                
            else:
                return {
                    'success': False,
                    'error': f'Unknown processing type: {processing_type}',
                    'response': None
                }
            
            # Add metadata
            generation_time = (datetime.now() - start_time).total_seconds()
            
            response_result.update({
                'query': query,
                'processing_type': processing_type,
                'generation_time': generation_time,
                'timestamp': datetime.now().isoformat(),
                'processing_success': processing_result.get('success', False)
            })
            
            # Add fallback response if generation failed but processing succeeded
            if not response_result['success'] and processing_result.get('success'):
                fallback_response = self.generate_fallback_response(query, processing_result)
                response_result.update({
                    'success': True,
                    'response': fallback_response,
                    'response_type': 'fallback'
                })
            
            return response_result
            
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'query': query,
                'generation_time': (datetime.now() - start_time).total_seconds()
            }
    
    def generate_off_topic_response(self, query: str) -> Dict[str, Any]:
        """Generate response for off-topic queries"""
        try:
            # Polite rejection message
            response = "I am FetiiGPT, specialized in analyzing Fetii Austin rideshare data. I can only answer questions about transportation patterns, trip data, user demographics, and group rideshare analytics. Please ask me something about Fetii transportation data instead!"
            
            return {
                'success': True,
                'response': response,
                'response_type': 'off_topic_rejection',
                'generation_time': 0.001
            }
            
        except Exception as e:
            logger.error(f"Error generating off-topic response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    def generate_fallback_response(self, query: str, processing_result: Dict[str, Any]) -> str:
        """Generate a simple fallback response when LLM generation fails"""
        try:
            processing_type = processing_result.get('processing_type', 'unknown')
            results = processing_result.get('results', {})
            
            if processing_type == 'SQL' and results.get('success'):
                data = results.get('data')
                if isinstance(data, pd.DataFrame):
                    return f"I found {len(data)} records in response to your query about: {query}. The data includes {len(data.columns)} columns: {', '.join(data.columns)}."
                else:
                    return f"I processed your SQL query about: {query}. The query executed successfully."
            
            elif processing_type == 'RAG' and results.get('success'):
                snippet_count = results.get('returned_results', 0)
                return f"I found {snippet_count} relevant records related to your query about: {query}."
            
            elif processing_type == 'HYBRID':
                sql_success = results.get('sql_result', {}).get('success', False)
                rag_success = results.get('rag_result', {}).get('success', False)
                return f"I processed your query about: {query} using both structured data analysis (SQL: {'successful' if sql_success else 'failed'}) and contextual search (RAG: {'successful' if rag_success else 'failed'})."
            
            else:
                return f"I processed your query about: {query}, but encountered an issue generating a detailed response. Please try rephrasing your question."
                
        except Exception as e:
            return f"I attempted to process your query about: {query}, but encountered technical difficulties. Please try again."

def main():
    """Test the response generator with sample data"""
    print("Testing Response Generation Layer")
    print("=" * 50)
    
    # Initialize response generator
    response_gen = ResponseGenerator()
    
    if not response_gen.openai_api_key:
        print("OpenAI API key not found. Cannot test response generation.")
        print("Please set OPENAI_API_KEY in your .env file.")
        return
    
    # Test SQL response generation
    print("\n1. Testing SQL Response Generation:")
    print("-" * 40)
    
    sample_sql_result = {
        'success': True,
        'sql_query': 'SELECT COUNT(*) as total_users FROM CustomerDemographics',
        'data': pd.DataFrame({'total_users': [150]}),
        'row_count': 1,
        'execution_time': 0.045,
        'data_type': 'dataframe'
    }
    
    sql_response = response_gen.generate_sql_response(
        query="How many users are in the database?", 
        sql_result=sample_sql_result
    )
    
    if sql_response['success']:
        print(f"SQL Response: {sql_response['response']}")
        print(f"Tokens used: {sql_response.get('tokens_used', 'N/A')}")
    else:
        print(f"SQL Response failed: {sql_response['error']}")
    
    # Test RAG response generation
    print("\n2. Testing RAG Response Generation:")
    print("-" * 40)
    
    sample_rag_result = {
        'success': True,
        'context_snippets': [
            {
                'rank': 1,
                'relevance_score': 0.85,
                'content': 'Customer profile: User ID: 168928, Age: 26 years old, Gender: Male',
                'type': 'demographics',
                'identifiers': 'UserID: 168928'
            },
            {
                'rank': 2,
                'relevance_score': 0.72,
                'content': 'Transportation trip: Trip ID: 726765, Pickup location: Downtown, Passengers: 5',
                'type': 'trip',
                'identifiers': 'TripID: 726765'
            }
        ],
        'total_results': 5,
        'returned_results': 2,
        'search_metadata': {'model_used': 'all-MiniLM-L6-v2'}
    }
    
    rag_response = response_gen.generate_rag_response(
        query="Tell me about user 168928",
        rag_result=sample_rag_result
    )
    
    if rag_response['success']:
        print(f"RAG Response: {rag_response['response']}")
        print(f"Tokens used: {rag_response.get('tokens_used', 'N/A')}")
    else:
        print(f"RAG Response failed: {rag_response['error']}")
    
    # Test fallback response
    print("\n3. Testing Fallback Response:")
    print("-" * 40)
    
    sample_processing_result = {
        'success': True,
        'processing_type': 'SQL',
        'results': sample_sql_result
    }
    
    fallback_response = response_gen.generate_fallback_response(
        query="How many users are in the database?",
        processing_result=sample_processing_result
    )
    
    print(f"Fallback Response: {fallback_response}")
    
    print(f"\n{'='*50}")
    print("Response Generation Testing Complete")

if __name__ == "__main__":
    main()