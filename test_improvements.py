from chatbot.hybrid_controller import HybridController
from chatbot.response_generator import ResponseGenerator

def test_improved_system():
    """Test the improved chatbot system for accuracy"""
    
    # Initialize components
    print("Initializing chatbot components...")
    hc = HybridController()
    rg = ResponseGenerator()
    
    # Test queries
    test_queries = [
        "How many groups went to Moody Center?",
        "What are the top drop-off spots for 18-24 year-olds on Saturday nights?", 
        "When do large groups (6+ riders) typically ride downtown?",
        "What's the average group size for Fetii rides?",
        "How many users are in the database?"
    ]
    
    print("\n" + "="*60)
    print("TESTING IMPROVED CHATBOT ACCURACY")
    print("="*60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}/5 ---")
        print(f"Query: {query}")
        
        try:
            # Process the query
            processing_result = hc.process_query(query)
            print(f"Processing Type: {processing_result.get('processing_type', 'unknown')}")
            print(f"Processing Success: {processing_result.get('success', False)}")
            
            # Generate response
            response_result = rg.generate_response(query, processing_result)
            print(f"Response Success: {response_result.get('success', False)}")
            
            if response_result.get('success'):
                response = response_result['response']
                print(f"Response Length: {len(response)} characters")
                print(f"Response: {response}")
            else:
                print(f"Error: {response_result.get('error', 'Unknown error')}")
                
            # Show SQL query if available
            if processing_result.get('processing_type') == 'SQL':
                sql_query = processing_result.get('results', {}).get('sql_query', 'No query')
                print(f"SQL Query: {sql_query}")
                
        except Exception as e:
            print(f"Error testing query: {e}")
        
        print("-" * 40)
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_improved_system()
