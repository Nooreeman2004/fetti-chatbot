import streamlit as st
import sys
import os
import pandas as pd
import json
from datetime import datetime
import logging
import traceback

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from chatbot.hybrid_controller import HybridController
    from chatbot.response_generator import ResponseGenerator
except ImportError:
    st.error("Could not import required modules. Please ensure all components are properly installed.")
    st.stop()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="FetiiGPT - Austin Group Rideshare Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-left: 4px solid #2196f3;
        color: #1a1a1a;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        background-color: #ffffff;
        border: 1px solid #4caf50;
        border-left: 4px solid #4caf50;
        color: #2e2e2e;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        color: #c62828;
    }
    
    .debug-info {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 0.8rem;
        font-family: monospace;
        font-size: 0.85rem;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Ensure text visibility in all elements */
    .stMarkdown, .stText, .stTextInput, .stTextArea {
        color: #2e2e2e !important;
    }
    
    /* Improve readability for bot responses */
    .bot-message p, .bot-message div, .bot-message span {
        color: #2e2e2e !important;
        font-weight: 400;
    }
    
    /* Ensure strong text is visible */
    .bot-message strong, .bot-message b {
        color: #1a1a1a !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class ChatbotUI:
    """Streamlit UI wrapper for the Transportation Chatbot"""
    
    def __init__(self):
        self.hybrid_controller = None
        self.response_generator = None
        self.initialization_status = {}
        
    def initialize_components(self):
        """Initialize the chatbot components with error handling"""
        try:
            # Initialize Hybrid Controller
            with st.spinner("Initializing chatbot components..."):
                self.hybrid_controller = HybridController()
                self.response_generator = ResponseGenerator()
                
            # Get system status
            self.initialization_status = self.hybrid_controller.get_system_status()
            
            # Check if components are healthy
            if self.initialization_status['overall_status'] == 'healthy':
                st.success("✅ All chatbot components initialized successfully!")
                return True
            else:
                st.warning("⚠️ Some components have issues. Chatbot may have limited functionality.")
                return True
                
        except Exception as e:
            st.error(f"❌ Failed to initialize chatbot: {str(e)}")
            logger.error(f"Initialization error: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def display_system_status(self):
        """Display system status in sidebar"""
        st.sidebar.header("🔧 System Status")
        
        if not self.initialization_status:
            st.sidebar.error("System not initialized")
            return
        
        # Overall status
        status = self.initialization_status['overall_status']
        if status == 'healthy':
            st.sidebar.success(f"Status: {status.upper()}")
        else:
            st.sidebar.warning(f"Status: {status.upper()}")
        
        # Component status
        st.sidebar.subheader("Components:")
        for component, info in self.initialization_status['components'].items():
            if info['status'] == 'success':
                st.sidebar.success(f"✅ {component.replace('_', ' ').title()}")
            else:
                st.sidebar.error(f"❌ {component.replace('_', ' ').title()}")
                if info['error']:
                    st.sidebar.error(f"Error: {info['error']}")
    
    def display_sample_queries(self):
        """Display sample queries in sidebar"""
        st.sidebar.header("💡 Sample Queries")
        
        sample_queries = [
            "How many groups went to Moody Center last month?",
            "What are the top drop-off spots for 18-24 year-olds on Saturday nights?",
            "When do large groups (6+ riders) typically ride downtown?",
            "What's the average group size for Fetii rides in Austin?",
            "Which Austin neighborhoods have the most Fetii activity?",
            "Show me Fetii trips with large groups (8+ passengers)",
            "Analyze Fetii user demographics and travel patterns",
            "What are the most common pickup locations for Fetii groups?",
            "How many Fetii users are in the Austin database?",
            "Compare Fetii usage patterns between different age groups"
        ]
        
        for i, query in enumerate(sample_queries, 1):
            if st.sidebar.button(f"{i}. {query}", key=f"sample_{i}"):
                st.session_state.user_input = query
                st.rerun()
    
    def process_user_query(self, query: str, show_debug: bool = False):
        """Process user query and display results"""
        if not query.strip():
            st.error("Please enter a query.")
            return
        
        # Display user message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {query}
        </div>
        """, unsafe_allow_html=True)
        
        # Process query
        with st.spinner("Processing your query..."):
            try:
                # Get processing results from hybrid controller
                processing_result = self.hybrid_controller.process_query(query)
                
                # Generate natural language response
                response_result = self.response_generator.generate_response(query, processing_result)
                
                # Display bot response
                if response_result['success']:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>Chatbot:</strong><br>{response_result['response']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Display fallback response
                    fallback = self.response_generator.generate_fallback_response(query, processing_result)
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>Chatbot:</strong><br>{fallback}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if show_debug:
                        st.error(f"Response generation error: {response_result.get('error', 'Unknown error')}")
                
                # Display debug information if requested
                if show_debug:
                    self.display_debug_info(processing_result, response_result)
                    
                # Store in chat history
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'query': query,
                    'processing_result': processing_result,
                    'response_result': response_result
                })
                
            except Exception as e:
                st.markdown(f"""
                <div class="chat-message error-message">
                    <strong>Error:</strong> {str(e)}
                </div>
                """, unsafe_allow_html=True)
                logger.error(f"Query processing error: {e}")
                logger.error(traceback.format_exc())
    
    def display_debug_info(self, processing_result, response_result):
        """Display detailed debug information"""
        st.subheader("🔍 Debug Information")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Processing Details")
            
            # Processing metrics
            metrics_data = {
                'Processing Type': processing_result.get('processing_type', 'Unknown'),
                'Classification Confidence': f"{processing_result.get('classification', {}).get('confidence', 0):.2f}",
                'Processing Time': f"{processing_result.get('processing_time', 0):.3f}s",
                'Success': '✅ Yes' if processing_result.get('success') else '❌ No'
            }
            
            for key, value in metrics_data.items():
                st.metric(key, value)
            
            # SQL Query (if applicable)
            results = processing_result.get('results', {})
            if processing_result.get('processing_type') in ['SQL', 'HYBRID']:
                sql_result = results if processing_result.get('processing_type') == 'SQL' else results.get('sql_result', {})
                if sql_result.get('sql_query'):
                    st.subheader("🗄️ SQL Query")
                    st.code(sql_result['sql_query'], language='sql')
                    
                    # SQL Results
                    if sql_result.get('success') and sql_result.get('data') is not None:
                        st.subheader("📊 SQL Results")
                        data = sql_result['data']
                        if isinstance(data, pd.DataFrame):
                            if len(data) <= 50:  # Show full data for small results
                                st.dataframe(data)
                            else:  # Show sample for large results
                                st.info(f"Showing first 50 rows of {len(data)} total rows")
                                st.dataframe(data.head(50))
                        else:
                            st.json(data)
        
        with col2:
            st.subheader("Response Generation")
            
            # Response metrics
            response_metrics = {
                'Response Type': response_result.get('response_type', 'Unknown'),
                'Generation Time': f"{response_result.get('generation_time', 0):.3f}s",
                'Tokens Used': response_result.get('tokens_used', 'N/A'),
                'Success': '✅ Yes' if response_result.get('success') else '❌ No'
            }
            
            for key, value in response_metrics.items():
                st.metric(key, value)
            
            # RAG Context (if applicable)
            if processing_result.get('processing_type') in ['RAG', 'HYBRID']:
                rag_result = results if processing_result.get('processing_type') == 'RAG' else results.get('rag_result', {})
                if rag_result.get('success') and rag_result.get('context_snippets'):
                    st.subheader("🔍 Retrieved Context")
                    
                    for i, snippet in enumerate(rag_result['context_snippets'][:3], 1):  # Show top 3
                        with st.expander(f"Context {i} (Score: {snippet.get('relevance_score', 0):.3f})"):
                            st.write(f"**Type:** {snippet.get('type', 'Unknown')}")
                            st.write(f"**Content:** {snippet.get('content', '')}")
                            if snippet.get('identifiers'):
                                st.write(f"**Identifiers:** {snippet['identifiers']}")
        
        # Full results JSON (collapsible)
        with st.expander("📋 Full Processing Results (JSON)"):
            st.json({
                'processing_result': processing_result,
                'response_result': response_result
            })
    
    def display_chat_history(self):
        """Display chat history"""
        if 'chat_history' not in st.session_state or not st.session_state.chat_history:
            st.info("No chat history yet. Start by asking a question!")
            return
        
        st.subheader("💬 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:]), 1):  # Show last 10
            with st.expander(f"Query {len(st.session_state.chat_history) - i + 1}: {chat['query'][:50]}..."):
                st.write(f"**Time:** {chat['timestamp']}")
                st.write(f"**Query:** {chat['query']}")
                
                if chat['response_result'].get('success'):
                    st.write(f"**Response:** {chat['response_result']['response']}")
                
                st.write(f"**Processing Type:** {chat['processing_result'].get('processing_type')}")
                st.write(f"**Success:** {'Yes' if chat['processing_result'].get('success') else 'No'}")
    
    def run(self):
        """Main Streamlit app runner"""
        # Header
        st.markdown('<h1 class="main-header">🚗 FetiiGPT - Austin Group Rideshare Analytics</h1>', unsafe_allow_html=True)
        st.markdown("Ask questions about Fetii's group rideshare data, Austin transportation patterns, and user analytics!")
        
        # Initialize components if not done
        if self.hybrid_controller is None:
            if not self.initialize_components():
                st.stop()
        
        # Sidebar
        with st.sidebar:
            self.display_system_status()
            st.divider()
            self.display_sample_queries()
            
            # Settings
            st.header("⚙️ Settings")
            show_debug = st.checkbox("Show debug information", value=False)
            
            if st.button("🗑️ Clear Chat History"):
                if 'chat_history' in st.session_state:
                    del st.session_state.chat_history
                st.success("Chat history cleared!")
        
        # Main chat interface
        st.header("💬 Chat Interface")
        
        # Query input
        user_query = st.text_input(
            "Ask me anything about the transportation data:",
            value=st.session_state.get('user_input', ''),
            placeholder="e.g., How many users are in the database?",
            key="query_input"
        )
        
        # Clear the session state input after using it
        if 'user_input' in st.session_state:
            del st.session_state.user_input
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            submit_button = st.button("🚀 Submit Query", type="primary")
        
        with col2:
            clear_button = st.button("🧹 Clear")
        
        # Process query
        if submit_button and user_query:
            self.process_user_query(user_query, show_debug)
        elif clear_button:
            st.session_state.query_input = ""
            st.experimental_rerun()
        
        # Chat history section
        st.divider()
        self.display_chat_history()

def main():
    """Main function to run the Streamlit app"""
    try:
        # Initialize and run the chatbot UI
        chatbot_ui = ChatbotUI()
        chatbot_ui.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        
        # Show troubleshooting tips
        st.subheader("🔧 Troubleshooting")
        st.markdown("""
        If you're seeing this error, try the following:
        
        1. **Check Dependencies**: Ensure all required packages are installed:
           ```bash
           pip install -r requirements.txt
           ```
        
        2. **Database Setup**: Make sure your database exists:
           ```bash
           python seed_data.py
           ```
        
        3. **Embeddings**: Ensure FAISS embeddings are built:
           ```bash
           python build_embeddings.py
           ```
        
        4. **Environment Variables**: Check your `.env` file contains:
           ```
           OPENAI_API_KEY=your-key-here
           PINECONE_API_KEY=your-pinecone-key-here
           ```
        
        5. **File Structure**: Verify all Python files are in the correct locations.
        
        6. **Manual Setup**: If automatic setup fails, you can run:
           ```bash
           python database/seed_data.py
           python embeddings/build_openai_pinecone_embeddings.py
           ```
           
        **Note**: Database and Pinecone embeddings are now automatically initialized when you run the app!
        
        7. **Pinecone Setup**: Make sure your Pinecone index exists and you have valid API keys
        """)

if __name__ == "__main__":
    main()