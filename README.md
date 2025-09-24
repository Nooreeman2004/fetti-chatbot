# Transportation Data Chatbot 🚗

A sophisticated hybrid chatbot system that combines SQL query processing, RAG (Retrieval-Augmented Generation), and natural language understanding to provide intelligent insights from transportation data.

## 🌟 Features

- Hybrid Query Processing: Automatically classifies and routes queries to SQL, RAG, or hybrid processing
- **Natural Language to SQL**: Converts natural language questions into SQL queries using OpenAI GPT
- **Semantic Search**: RAG-based retrieval for contextual information and entity descriptions
- **Interactive Web UI**: Modern Streamlit-based interface with real-time chat
- **Comprehensive Analytics**: Support for complex transportation data analysis
- **Robust Error Handling**: Graceful fallbacks and detailed error reporting

## 🏗️ Architecture

The system follows a modular architecture with the following key components:

### Core Components

1. **Query Classifier** (`chatbot/query_classifier.py`)
   - Rule-based and LLM-powered query classification
   - Routes queries to SQL, RAG, or hybrid processing
   - Confidence-based decision making

2. **SQL Engine** (`chatbot/sql_engine.py`)
   - Natural language to SQL translation
   - Query validation and execution
   - Schema-aware query generation

3. **RAG Engine** (`chatbot/rag_engine.py`)
   - FAISS-based vector similarity search
   - Sentence transformer embeddings
   - Contextual information retrieval

4. **Hybrid Controller** (`chatbot/hybrid_controller.py`)
   - Orchestrates SQL and RAG processing
   - Combines structured and unstructured data
   - Manages complex multi-step queries

5. **Response Generator** (`chatbot/response_generator.py`)
   - Converts query results to natural language
   - Formats responses with context and insights
   - Handles different response types

### Data Layer

- **Database**: SQLite with transportation data schema
- **Vector Store**: FAISS index for semantic search
- **CSV Data**: Customer demographics, trip data, and check-in records

## 📊 Data Schema

The system works with three main data tables:

### CustomerDemographics
- `UserID` (Primary Key)
- `Age`, `Gender`, `Name`
- `CreatedAt` timestamp

### TripData
- `TripID` (Primary Key)
- `BookingUserID` (Foreign Key)
- Pickup/Dropoff coordinates and addresses
- `TripDate`, `TotalPassengers`, `Duration`, `Distance`

### CheckedInUsers
- `CheckInID` (Primary Key)
- `UserID`, `TripID` (Foreign Keys)
- `CheckInTime`, `CheckInStatus`

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (for NL-to-SQL and response generation)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Fetii-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Initialize the database**
   ```bash
   python database/seed_data.py
   ```

5. **Build embeddings for RAG**
   ```bash
   python embeddings/build_embeddings.py
   ```

6. **Launch the application**
   ```bash
   streamlit run ui/app.py
   ```

## 📁 Project Structure

```
Fetii-chatbot/
├── chatbot/                    # Core chatbot modules
│   ├── hybrid_controller.py   # Main orchestration layer
│   ├── query_classifier.py    # Query classification logic
│   ├── rag_engine.py         # RAG processing engine
│   ├── response_generator.py # Response formatting
│   └── sql_engine.py         # SQL query processing
├── config/                    # Configuration files
│   ├── logging.conf          # Logging configuration
│   └── settings.yaml         # Application settings
├── data/                      # Raw data files
│   ├── CheckedIn_UserID's.csv
│   ├── CustomerDemographics.csv
│   └── TripData.csv
├── database/                  # Database management
│   ├── chatbot.db            # SQLite database
│   ├── schema.sql            # Database schema
│   └── seed_data.py          # Data seeding script
├── embeddings/               # Vector embeddings
│   ├── build_embeddings.py   # Embedding generation
│   └── vectordb/             # FAISS index storage
├── notebooks/                # Jupyter notebooks
│   ├── data_exploration.ipynb
│   └── query_examples.ipynb
├── tests/                    # Test suite
│   ├── test_hybrid_engine.py
│   ├── test_rag_engine.py
│   └── test_sql_engine.py
├── ui/                       # User interface
│   └── app.py               # Streamlit application
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for natural language processing features

### Settings (`config/settings.yaml`)

```yaml
database_path: "./database/chatbot.db"
embeddings_path: "./embeddings/vectordb/faiss_index.pkl"
```

## 💬 Usage Examples

### SQL Queries
- "How many trips were taken last month?"
- "What's the average trip duration?"
- "Show me the top 5 users by trip count"

### RAG Queries
- "Tell me about user 12345"
- "Describe the transportation patterns"
- "Explain how the booking system works"

### Hybrid Queries
- "Show me the top 5 users by trip count and their profiles"
- "Compare trip durations between different age groups"
- "Find users over 30 and describe their travel behavior"

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_sql_engine.py -v
pytest tests/test_rag_engine.py -v
pytest tests/test_hybrid_engine.py -v
```

## 📈 Performance

- **Query Classification**: Rule-based with LLM fallback
- **SQL Processing**: Optimized with indexes and query validation
- **RAG Retrieval**: FAISS-based similarity search with configurable top-k
- **Response Generation**: OpenAI GPT-3.5-turbo for natural language responses

## 🔍 Monitoring & Logging

The system includes comprehensive logging:

- Query classification decisions
- SQL query execution results
- RAG retrieval performance
- Error tracking and debugging information

Logs are configured via `config/logging.conf` and can be viewed in the Streamlit interface.

## 🛠️ Development

### Adding New Query Types

1. Update patterns in `query_classifier.py`
2. Add processing logic in `hybrid_controller.py`
3. Update response templates in `response_generator.py`

### Extending the Schema

1. Update `database/schema.sql`
2. Modify data processing in `database/seed_data.py`
3. Update schema context in `sql_engine.py`

### Customizing Embeddings

1. Modify model selection in `embeddings/build_embeddings.py`
2. Update embedding dimensions in `rag_engine.py`
3. Rebuild the FAISS index

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the test suite for usage examples
2. Review the Jupyter notebooks for data exploration
3. Examine the logging output for debugging information
4. Create an issue with detailed error information

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Real-time data streaming
- [ ] Custom model fine-tuning
- [ ] API endpoint development
- [ ] Docker containerization
- [ ] Cloud deployment support

---

**Built with ❤️ for intelligent transportation data analysis**
"# Fetii" 
