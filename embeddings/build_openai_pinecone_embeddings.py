import pandas as pd
import numpy as np
import os
import json
from sqlalchemy import create_engine, text
from datetime import datetime
import logging
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIPineconeEmbeddingLayer:
    def __init__(self, 
                 db_path: str = 'database/transportation.db',
                 pinecone_api_key: str = None,
                 openai_api_key: str = None,
                 index_name: str = 'fetii-chatbot',
                 environment: str = 'us-east-1'):
        """
        Initialize the OpenAI + Pinecone embedding preparation layer
        
        Args:
            db_path: Path to SQLite database
            pinecone_api_key: Pinecone API key
            openai_api_key: OpenAI API key
            index_name: Pinecone index name
            environment: Pinecone environment
        """
        self.db_path = db_path
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'fetii-chatbot')
        self.environment = environment or os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        
        # Initialize components
        self.pc = None
        self.index = None
        self.openai_client = None
        self.metadata = []
        self.text_representations = []
        
        # Initialize clients
        self.initialize_pinecone()
        self.initialize_openai()
        
    def initialize_pinecone(self):
        """Initialize Pinecone client and index"""
        try:
            from pinecone import Pinecone
            
            logger.info("Initializing Pinecone client...")
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists, create if not
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric='cosine',
                    spec={
                        'serverless': {
                            'cloud': 'aws',
                            'region': self.environment
                        }
                    }
                )
                # Wait for index to be ready
                time.sleep(60)
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
            
            self.index = self.pc.Index(self.index_name)
            logger.info("Pinecone initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
    
    def initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            
            logger.info("Initializing OpenAI client...")
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
            raise
    
    def get_openai_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            raise
    
    def get_openai_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Get embeddings from OpenAI in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch_texts
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Processed embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                raise
        
        return embeddings
    
    def load_data_from_database(self) -> Dict[str, pd.DataFrame]:
        """Load data from SQLite database tables"""
        try:
            engine = create_engine(f'sqlite:///{self.db_path}')
            
            # Load all tables
            with engine.connect() as conn:
                # Customer Demographics
                demographics_query = "SELECT * FROM CustomerDemographics"
                df_demographics = pd.read_sql(demographics_query, conn)
                
                # Trip Data
                trip_data_query = "SELECT * FROM TripData"
                df_trip_data = pd.read_sql(trip_data_query, conn)
                
                # Checked In Users
                checkin_query = "SELECT * FROM CheckedInUsers"
                df_checkin = pd.read_sql(checkin_query, conn)
                
                # Combined view for comprehensive embeddings
                combined_query = """
                SELECT 
                    cd.UserID,
                    cd.Age,
                    cd.Gender,
                    cd.Name,
                    td.TripID,
                    td.PickUpLatitude,
                    td.PickUpLongitude,
                    td.DropOffLatitude,
                    td.DropOffLongitude,
                    td.PickUpAddress,
                    td.DropOffAddress,
                    td.TripDate,
                    td.TotalPassengers,
                    td.Duration,
                    td.Distance,
                    ci.CheckInTime,
                    ci.CheckInStatus
                FROM CustomerDemographics cd
                LEFT JOIN CheckedInUsers ci ON cd.UserID = ci.UserID
                LEFT JOIN TripData td ON ci.TripID = td.TripID
                """
                df_combined = pd.read_sql(combined_query, conn)
            
            logger.info(f"Loaded data - Demographics: {len(df_demographics)}, "
                       f"Trips: {len(df_trip_data)}, "
                       f"Check-ins: {len(df_checkin)}, "
                       f"Combined: {len(df_combined)}")
            
            return {
                'demographics': df_demographics,
                'trip_data': df_trip_data,
                'checkin': df_checkin,
                'combined': df_combined
            }
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise
    
    def convert_demographics_to_text(self, row: pd.Series) -> str:
        """Convert demographics row to descriptive text"""
        components = []
        
        if pd.notna(row.get('UserID')):
            components.append(f"User ID: {int(row['UserID'])}")
        
        if pd.notna(row.get('Age')):
            components.append(f"Age: {int(row['Age'])} years old")
        
        if pd.notna(row.get('Gender')) and row['Gender'] != 'Unknown':
            components.append(f"Gender: {row['Gender']}")
        
        if pd.notna(row.get('Name')) and not row['Name'].startswith('User_'):
            components.append(f"Name: {row['Name']}")
        
        return "Customer profile: " + ", ".join(components)
    
    def convert_trip_to_text(self, row: pd.Series) -> str:
        """Convert trip data row to descriptive text"""
        components = []
        
        if pd.notna(row.get('TripID')):
            components.append(f"Trip ID: {int(row['TripID'])}")
        
        if pd.notna(row.get('PickUpAddress')):
            pickup = row['PickUpAddress'].replace('#########', '').strip()
            if pickup:
                components.append(f"Pickup location: {pickup}")
        
        if pd.notna(row.get('DropOffAddress')):
            dropoff = row['DropOffAddress'].replace('#########', '').strip()
            if dropoff:
                components.append(f"Destination: {dropoff}")
        
        if pd.notna(row.get('TotalPassengers')):
            components.append(f"Passengers: {int(row['TotalPassengers'])}")
        
        if pd.notna(row.get('Duration')):
            components.append(f"Duration: {row['Duration']:.1f} minutes")
        
        if pd.notna(row.get('Distance')):
            components.append(f"Distance: {row['Distance']:.1f} km")
        
        if pd.notna(row.get('TripDate')):
            components.append(f"Date: {row['TripDate']}")
        
        # Add coordinates for location context
        if pd.notna(row.get('PickUpLatitude')) and pd.notna(row.get('PickUpLongitude')):
            components.append(f"Pickup coordinates: ({row['PickUpLatitude']:.4f}, {row['PickUpLongitude']:.4f})")
        
        return "Transportation trip: " + ", ".join(components)
    
    def convert_combined_to_text(self, row: pd.Series) -> str:
        """Convert combined row to comprehensive descriptive text"""
        components = []
        
        # User information
        if pd.notna(row.get('UserID')):
            user_info = [f"User ID: {int(row['UserID'])}"]
            
            if pd.notna(row.get('Age')):
                user_info.append(f"{int(row['Age'])} years old")
            
            if pd.notna(row.get('Gender')) and row['Gender'] != 'Unknown':
                user_info.append(f"{row['Gender']}")
            
            components.append("User: " + ", ".join(user_info))
        
        # Trip information
        if pd.notna(row.get('TripID')):
            trip_info = [f"Trip ID: {int(row['TripID'])}"]
            
            if pd.notna(row.get('PickUpAddress')):
                pickup = row['PickUpAddress'].replace('#########', '').strip()
                if pickup:
                    trip_info.append(f"from {pickup}")
            
            if pd.notna(row.get('DropOffAddress')):
                dropoff = row['DropOffAddress'].replace('#########', '').strip()
                if dropoff:
                    trip_info.append(f"to {dropoff}")
            
            if pd.notna(row.get('TotalPassengers')):
                trip_info.append(f"with {int(row['TotalPassengers'])} passengers")
            
            if pd.notna(row.get('Duration')):
                trip_info.append(f"duration {row['Duration']:.1f} minutes")
            
            components.append("Trip: " + ", ".join(trip_info))
        
        # Check-in information
        if pd.notna(row.get('CheckInTime')):
            checkin_info = [f"checked in at {row['CheckInTime']}"]
            
            if pd.notna(row.get('CheckInStatus')):
                checkin_info.append(f"status: {row['CheckInStatus']}")
            
            components.append("Check-in: " + ", ".join(checkin_info))
        
        return "Transportation record: " + " | ".join(components)
    
    def prepare_embeddings_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[List[str], List[Dict]]:
        """
        Prepare text representations and metadata for embedding generation
        
        Returns:
            Tuple of (text_representations, metadata_list)
        """
        text_representations = []
        metadata_list = []
        
        # Process demographics data
        for idx, row in data['demographics'].iterrows():
            text = self.convert_demographics_to_text(row)
            metadata = {
                'type': 'demographics',
                'id': f"demo_{row['UserID']}",
                'user_id': int(row['UserID']) if pd.notna(row['UserID']) else 0,
                'source_table': 'CustomerDemographics',
                'age': int(row['Age']) if pd.notna(row['Age']) else 0,
                'gender': str(row['Gender']) if pd.notna(row['Gender']) else 'Unknown',
                'name': str(row['Name']) if pd.notna(row['Name']) else 'Unknown'
            }
            text_representations.append(text)
            metadata_list.append(metadata)
        
        # Process trip data
        for idx, row in data['trip_data'].iterrows():
            text = self.convert_trip_to_text(row)
            metadata = {
                'type': 'trip',
                'id': f"trip_{row['TripID']}",
                'trip_id': int(row['TripID']) if pd.notna(row['TripID']) else 0,
                'booking_user_id': int(row.get('BookingUserID')) if pd.notna(row.get('BookingUserID')) else 0,
                'source_table': 'TripData',
                'total_passengers': int(row['TotalPassengers']) if pd.notna(row['TotalPassengers']) else 0,
                'duration': float(row['Duration']) if pd.notna(row['Duration']) else 0.0,
                'distance': float(row['Distance']) if pd.notna(row['Distance']) else 0.0
            }
            text_representations.append(text)
            metadata_list.append(metadata)
        
        # Process combined data for comprehensive search
        for idx, row in data['combined'].iterrows():
            if pd.notna(row.get('TripID')):  # Only include rows with trip data
                text = self.convert_combined_to_text(row)
                metadata = {
                    'type': 'combined',
                    'id': f"combined_{row['UserID']}_{row['TripID']}",
                    'user_id': int(row['UserID']) if pd.notna(row['UserID']) else 0,
                    'trip_id': int(row['TripID']) if pd.notna(row['TripID']) else 0,
                    'source_table': 'Combined',
                    'age': int(row['Age']) if pd.notna(row['Age']) else 0,
                    'gender': str(row['Gender']) if pd.notna(row['Gender']) else 'Unknown',
                    'total_passengers': int(row['TotalPassengers']) if pd.notna(row['TotalPassengers']) else 0
                }
                text_representations.append(text)
                metadata_list.append(metadata)
        
        logger.info(f"Prepared {len(text_representations)} text representations for embedding")
        return text_representations, metadata_list
    
    def upsert_to_pinecone(self, texts: List[str], metadata: List[Dict], embeddings: List[List[float]]):
        """Upsert embeddings to Pinecone"""
        try:
            logger.info(f"Upserting {len(texts)} embeddings to Pinecone...")
            
            # Prepare vectors for upsert
            vectors = []
            for i, (text, meta, embedding) in enumerate(zip(texts, metadata, embeddings)):
                vectors.append({
                    'id': meta['id'],
                    'values': embedding,
                    'metadata': {
                        **meta,
                        'text': text
                    }
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
                time.sleep(1)  # Rate limiting
            
            logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")
            
        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Pinecone index contains {stats['total_vector_count']} vectors")
            return {
                'total_vectors': stats['total_vector_count'],
                'index_name': self.index_name,
                'dimension': stats.get('dimension', 1536)
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {'error': str(e)}
    
    def save_config(self, total_embeddings: int):
        """Save configuration for the embedding setup"""
        try:
            config = {
                'embedding_model': 'text-embedding-3-small',
                'vector_db': 'pinecone',
                'total_embeddings': total_embeddings,
                'creation_timestamp': datetime.now().isoformat(),
                'db_path': self.db_path,
                'index_name': self.index_name,
                'environment': self.environment
            }
            
            # Create embeddings directory if it doesn't exist
            os.makedirs('embeddings', exist_ok=True)
            config_path = os.path.join('embeddings', 'openai_pinecone_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def build_embeddings_pipeline(self) -> Dict[str, Any]:
        """Execute the complete embedding preparation pipeline"""
        try:
            # Load data from database
            data = self.load_data_from_database()
            
            # Prepare text representations and metadata
            text_representations, metadata = self.prepare_embeddings_data(data)
            
            if not text_representations:
                raise ValueError("No text representations generated. Check your database data.")
            
            # Generate embeddings using OpenAI
            logger.info("Generating embeddings using OpenAI...")
            embeddings = self.get_openai_embeddings_batch(text_representations)
            
            # Upsert to Pinecone
            self.upsert_to_pinecone(text_representations, metadata, embeddings)
            
            # Get index statistics
            stats = self.get_index_stats()
            
            # Save configuration
            self.save_config(len(text_representations))
            
            # Store for later use
            self.metadata = metadata
            self.text_representations = text_representations
            
            return {
                'success': True,
                'total_embeddings': len(text_representations),
                'total_vectors': stats.get('total_vectors', 0),
                'index_name': stats.get('index_name', ''),
                'embedding_model': 'text-embedding-3-small'
            }
            
        except Exception as e:
            logger.error(f"Error in embedding pipeline: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar records using semantic similarity
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing search results
        """
        try:
            # Get query embedding
            query_embedding = self.get_openai_embedding(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for i, match in enumerate(results['matches']):
                result = {
                    'rank': i + 1,
                    'score': float(match['score']),
                    'text': match['metadata'].get('text', ''),
                    'metadata': {k: v for k, v in match['metadata'].items() if k != 'text'}
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise

def main():
    """Main function to execute embedding preparation"""
    # Initialize embedding preparation layer
    embedder = OpenAIPineconeEmbeddingLayer()
    
    # Build embeddings pipeline
    logger.info("Starting OpenAI + Pinecone embedding preparation pipeline...")
    result = embedder.build_embeddings_pipeline()
    
    if result['success']:
        logger.info("Embedding preparation completed successfully!")
        logger.info(f"Total embeddings: {result['total_embeddings']}")
        logger.info(f"Total vectors: {result['total_vectors']}")
        logger.info(f"Index name: {result['index_name']}")
        logger.info(f"Embedding model: {result['embedding_model']}")
        
        # Test search functionality
        test_queries = [
            "young male user from Austin",
            "trip with 10 passengers",
            "user aged 25 years"
        ]
        
        logger.info("\nTesting semantic search:")
        for query in test_queries:
            try:
                results = embedder.search_similar(query, top_k=3)
                logger.info(f"\nQuery: '{query}'")
                for result in results:
                    logger.info(f"  Score: {result['score']:.4f} | {result['text'][:100]}...")
            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
    
    else:
        logger.error(f"Embedding preparation failed: {result['error']}")

if __name__ == "__main__":
    main()
