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
        return """You are FetiiGPT, an intelligent transportation data analyst specialized in Fetii's group rideshare data from Austin, Texas. You provide accurate, helpful insights about group transportation patterns, user behavior, and Austin rideshare trends.

Your expertise includes:
- Fetii's group rideshare operations in Austin
- User demographics and travel patterns  
- Popular pickup/dropoff locations and routes
- Group size trends and peak usage times
- Transportation efficiency and user preferences
- Real-world insights about Austin's transportation landscape

Key guidelines:
1. Always be conversational and friendly while being accurate about Fetii data
2. Present data clearly with specific numbers and context
3. Validate data quality and mention any limitations (e.g., missing age data)
4. For empty results, explain why and suggest alternative queries
5. For large datasets, summarize key findings and highlight interesting patterns
6. When discussing specific users or trips, be respectful of privacy
7. Suggest follow-up questions when appropriate
8. If data seems inconsistent or incomplete, acknowledge this transparently
9. Reference Austin locations and landmarks when relevant to Fetii trips
10. Always ground your responses in the actual data provided

Response style:
- Use natural, conversational language
- Structure responses with clear paragraphs
- Use bullet points or numbered lists for multiple items when helpful
- Always cite specific data points to support your statements
- Be enthusiastic about helping users understand Fetii's transportation data
- Acknowledge data limitations honestly"""
    
    def _initialize_sql_prompt(self) -> str:
        """Initialize the prompt template for SQL-based responses"""
        return """Generate a direct, conversational response about Fetii's Austin group rideshare data.

User Question: {query}

Query Results: {sql_results}

Provide a natural language response that:
1. Directly answers the user's question with specific numbers
2. If no results found, simply state that no data matches the criteria
3. Highlight any interesting patterns or insights
4. Keep the response focused and conversational
5. Avoid technical details about data validation or execution

Response:"""
    
    def _initialize_rag_prompt(self) -> str:
        """Initialize the prompt template for RAG-based responses"""
        return """Provide a direct answer to the user's question about Fetii's rideshare data.

User Question: {query}

Context Information:
{context_snippets}

Please provide a natural language response that:
1. Directly addresses the user's question using the context provided
2. Synthesizes information from multiple sources when applicable
3. Highlights the most relevant details about Fetii operations
4. Keep the response focused and conversational
5. Avoid technical details about search metadata or results counts

Response:"""
    
    def _initialize_hybrid_prompt(self) -> str:
        """Initialize the prompt template for hybrid responses"""
        return """Provide a comprehensive response that combines both data sources about Fetii's rideshare operations.

User Question: {query}

SQL Results:
{sql_summary}

Context Information:
{rag_summary}

Please provide a natural language response that:
1. Directly answers the user's question using both data sources
2. Integrates quantitative and qualitative insights naturally
3. Highlights the most relevant findings about Fetii operations
4. Keep the response focused and conversational
5. Avoid technical details about data validation or processing

Response:"""
    
    def validate_sql_results(self, sql_result: Dict[str, Any]) -> Dict[str, str]:
        """Validate SQL results and generate quality notes"""
        quality_notes = []
        
        if not sql_result.get('success'):
            return {'data_quality_notes': f"Query failed: {sql_result.get('error', 'Unknown error')}"}
        
        data = sql_result.get('data')
        row_count = sql_result.get('row_count', 0)
        
        # Check for empty results
        if row_count == 0:
            quality_notes.append("No matching records found")
        
        # Check for data quality issues if DataFrame
        if isinstance(data, pd.DataFrame) and len(data) > 0:
            # Check for null values
            null_counts = data.isnull().sum()
            if null_counts.any():
                null_cols = [col for col, count in null_counts.items() if count > 0]
                quality_notes.append(f"Some null values found in columns: {', '.join(null_cols)}")
            
            # Check for age-related queries with missing data
            if 'Age' in data.columns:
                null_ages = data['Age'].isnull().sum()
                if null_ages > 0:
                    quality_notes.append(f"{null_ages} records have missing age data")
            
            # Check for reasonable data ranges
            if 'TotalPassengers' in data.columns:
                max_passengers = data['TotalPassengers'].max()
                if max_passengers > 20:
                    quality_notes.append(f"Some unusually large groups found (max: {max_passengers} passengers)")
        
        return {'data_quality_notes': '; '.join(quality_notes) if quality_notes else 'Data appears complete'}
    
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
                else:
                    sql_summary = f"SQL query executed but returned no results: {hybrid_result['sql_result'].get('error', 'Empty result set')}"
            
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
                else:
                    rag_summary = "RAG search completed but found no relevant context."
            
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
    
    def generate_sql_response(self, query: str, sql_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for SQL-only results"""
        if not self.openai_api_key:
            return {
                'success': False,
                'error': 'OpenAI API key not available',
                'response': None
            }
        
        try:
            # Validate and format SQL results
            validation = self.validate_sql_results(sql_result)
            formatted_results = self.format_sql_results(sql_result)
            
            # Prepare prompt
            prompt = self.sql_prompt_template.format(
                query=query,
                sql_query=sql_result.get('sql_query', 'Query not available'),
                sql_results=formatted_results,
                row_count=sql_result.get('row_count', 0),
                execution_time=sql_result.get('execution_time', 0),
                data_type=sql_result.get('data_type', 'unknown'),
                data_quality_notes=validation['data_quality_notes']
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
                'tokens_used': response.usage.total_tokens,
                'generation_time': 0  # Could add timing if needed
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    def generate_rag_response(self, query: str, rag_result: Dict[str, Any]) -> Dict[str, Any]:
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
                'tokens_used': response.usage.total_tokens,
                'generation_time': 0
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    def generate_hybrid_response(self, query: str, hybrid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for hybrid results"""
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
                'tokens_used': response.usage.total_tokens,
                'generation_time': 0
            }
            
        except Exception as e:
            logger.error(f"Error generating hybrid response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    def generate_response(self, query: str, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to generate response based on processing results
        
        Args:
            query: User's original query
            processing_result: Results from hybrid controller
            
        Returns:
            Dictionary with generated response
        """
        start_time = datetime.now()
        
        try:
            processing_type = processing_result.get('processing_type')
            results = processing_result.get('results', {})
            
            if processing_type == 'SQL':
                response_result = self.generate_sql_response(query, results)
            elif processing_type == 'RAG':
                response_result = self.generate_rag_response(query, results)
            elif processing_type == 'HYBRID':
                response_result = self.generate_hybrid_response(query, results)
            else:
                return {
                    'success': False,
                    'error': f'Unknown processing type: {processing_type}',
                    'response': None
                }
            
            # Add generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            response_result['generation_time'] = generation_time
            
            return response_result
            
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'generation_time': (datetime.now() - start_time).total_seconds()
            }
    
    def generate_fallback_response(self, query: str, processing_result: Dict[str, Any]) -> str:
        """Generate a fallback response when main generation fails"""
        try:
            processing_type = processing_result.get('processing_type', 'unknown')
            error = processing_result.get('error', 'Unknown error')
            
            fallback = f"I apologize, but I encountered an issue processing your query about Fetii's rideshare data.\n\n"
            
            if processing_type == 'SQL':
                fallback += f"The database query failed with error: {error}\n\n"
                fallback += "This might be due to:\n"
                fallback += "- Complex query requirements\n"
                fallback += "- Data availability issues\n"
                fallback += "- Temporary system issues\n\n"
                fallback += "Please try rephrasing your question or asking about a different aspect of the Fetii data."
            
            elif processing_type == 'RAG':
                fallback += f"The context search failed with error: {error}\n\n"
                fallback += "This might be due to:\n"
                fallback += "- No relevant information found\n"
                fallback += "- Search index issues\n"
                fallback += "- Query complexity\n\n"
                fallback += "Please try asking about specific users, trips, or general Fetii patterns."
            
            elif processing_type == 'HYBRID':
                fallback += f"The combined analysis failed with error: {error}\n\n"
                fallback += "Please try asking a more specific question about either:\n"
                fallback += "- Numerical data (counts, averages, trends)\n"
                fallback += "- Specific user or trip information\n"
            
            else:
                fallback += f"Query processing failed: {error}\n\n"
                fallback += "Please try asking a simpler question about Fetii's Austin rideshare data."
            
            return fallback
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try your question again."

def main():
    """Test the Response Generator functionality"""
    print("Testing Response Generator")
    print("=" * 60)
    
    # Initialize generator
    generator = ResponseGenerator()
    
    # Test with mock data
    mock_sql_result = {
        'success': True,
        'data': pd.DataFrame({
            'DropOffAddress': ['Moody Center', 'Downtown Austin', '6th Street'],
            'trip_count': [15, 23, 8]
        }),
        'row_count': 3,
        'execution_time': 0.045,
        'data_type': 'dataframe',
        'sql_query': 'SELECT DropOffAddress, COUNT(*) as trip_count FROM TripData GROUP BY DropOffAddress'
    }
    
    mock_processing_result = {
        'processing_type': 'SQL',
        'results': mock_sql_result,
        'success': True
    }
    
    test_query = "What are the most popular drop-off locations?"
    
    print(f"Testing query: '{test_query}'")
    print("-" * 40)
    
    response_result = generator.generate_response(test_query, mock_processing_result)
    
    print(f"Success: {response_result['success']}")
    if response_result['success']:
        print(f"Response: {response_result['response']}")
        print(f"Tokens used: {response_result.get('tokens_used', 'N/A')}")
        print(f"Generation time: {response_result.get('generation_time', 0):.3f}s")
    else:
        print(f"Error: {response_result['error']}")
        print(f"Fallback: {generator.generate_fallback_response(test_query, mock_processing_result)}")

if __name__ == "__main__":
    main()
