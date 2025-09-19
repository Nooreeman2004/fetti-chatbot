import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
import os
from sqlalchemy import create_engine, text
from datetime import datetime
import logging
from typing import List, Dict, Any, Tuple
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingPreparationLayer:
    def __init__(self, 
                 db_path='database/transportation.db',
                 model_name='all-MiniLM-L6-v2',
                 chroma_persist_dir='embeddings/chromadb'):
        """
        Initialize the embedding preparation layer
        
        Args:
            db_path: Path to SQLite database
            model_name: SentenceTransformers model name
            chroma_persist_dir: Directory to save ChromaDB persistent storage
        """
        self.db_path = db_path
        self.model_name = model_name
        self.chroma_persist_dir = chroma_persist_dir
        self.model = None
        self.chroma_client = None
        self.collection = None
        self.metadata = []
        self.text_representations = []
        
        # Ensure embeddings directory exists
        self.ensure_embeddings_directory()
        
    def ensure_embeddings_directory(self):
        """Create embeddings directory if it doesn't exist"""
        if not os.path.exists(self.chroma_persist_dir):
            os.makedirs(self.chroma_persist_dir)
            logger.info(f"Created ChromaDB directory: {self.chroma_persist_dir}")
    
    def initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            logger.info("Initializing ChromaDB client...")
            
            # Initialize ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            collection_name = "transportation_embeddings"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except Exception as e:
                # Collection doesn't exist, create it
                logger.info(f"Collection not found ({type(e).__name__}), creating new collection: {collection_name}")
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Transportation data embeddings"}
                )
                logger.info(f"Created new collection: {collection_name}")
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def load_sentence_transformer(self):
        """Load the SentenceTransformer model"""
        try:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {e}")
            raise
    
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
    
    def add_embeddings_to_chromadb(self, texts: List[str], metadata: List[Dict]) -> None:
        """Add embeddings to ChromaDB collection"""
        try:
            logger.info(f"Adding {len(texts)} embeddings to ChromaDB...")
            
            # Generate embeddings in smaller batches to avoid ChromaDB limits
            embedding_batch_size = 100
            chromadb_batch_size = 1000  # Smaller batch for ChromaDB insertion
            
            all_embeddings = []
            all_ids = []
            all_metadatas = []
            
            # First, generate all embeddings
            for i in range(0, len(texts), embedding_batch_size):
                batch_texts = texts[i:i + embedding_batch_size]
                batch_metadata = metadata[i:i + embedding_batch_size]
                
                # Generate embeddings for this batch
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
                all_embeddings.extend(batch_embeddings.tolist())
                
                # Prepare IDs and metadata for this batch
                for j, meta in enumerate(batch_metadata):
                    all_ids.append(meta['id'])
                    all_metadatas.append(meta)
                
                logger.info(f"Processed embedding batch {i//embedding_batch_size + 1}/{(len(texts)-1)//embedding_batch_size + 1}")
            
            # Now add to ChromaDB in smaller batches
            for i in range(0, len(texts), chromadb_batch_size):
                batch_end = min(i + chromadb_batch_size, len(texts))
                batch_texts = texts[i:batch_end]
                batch_embeddings = all_embeddings[i:batch_end]
                batch_metadatas = all_metadatas[i:batch_end]
                batch_ids = all_ids[i:batch_end]
                
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                logger.info(f"Added ChromaDB batch {i//chromadb_batch_size + 1}/{(len(texts)-1)//chromadb_batch_size + 1}")
            
            logger.info(f"Successfully added {len(texts)} embeddings to ChromaDB")
            
        except Exception as e:
            logger.error(f"Error adding embeddings to ChromaDB: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics"""
        try:
            count = self.collection.count()
            logger.info(f"ChromaDB collection contains {count} documents")
            return {
                'total_documents': count,
                'collection_name': self.collection.name,
                'persist_directory': self.chroma_persist_dir
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def save_config(self, total_documents: int):
        """Save configuration for ChromaDB setup"""
        try:
            config = {
                'model_name': self.model_name,
                'total_documents': total_documents,
                'creation_timestamp': datetime.now().isoformat(),
                'db_path': self.db_path,
                'chroma_persist_dir': self.chroma_persist_dir,
                'collection_name': 'transportation_embeddings'
            }
            
            config_path = os.path.join(self.chroma_persist_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def build_embeddings_pipeline(self) -> Dict[str, Any]:
        """Execute the complete embedding preparation pipeline"""
        try:
            # Initialize ChromaDB
            self.initialize_chromadb()
            
            # Load SentenceTransformer model
            self.load_sentence_transformer()
            
            # Load data from database
            data = self.load_data_from_database()
            
            # Prepare text representations and metadata
            text_representations, metadata = self.prepare_embeddings_data(data)
            
            if not text_representations:
                raise ValueError("No text representations generated. Check your database data.")
            
            # Add embeddings to ChromaDB
            self.add_embeddings_to_chromadb(text_representations, metadata)
            
            # Get collection statistics
            stats = self.get_collection_stats()
            
            # Save configuration
            self.save_config(len(text_representations))
            
            # Store for later use
            self.metadata = metadata
            self.text_representations = text_representations
            
            return {
                'success': True,
                'total_embeddings': len(text_representations),
                'total_documents': stats.get('total_documents', 0),
                'collection_name': stats.get('collection_name', ''),
                'output_directory': self.chroma_persist_dir
            }
            
        except Exception as e:
            logger.error(f"Error in embedding pipeline: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def load_existing_embeddings(self) -> bool:
        """Load existing ChromaDB collection"""
        try:
            # Initialize ChromaDB
            self.initialize_chromadb()
            
            # Check if collection has data
            count = self.collection.count()
            if count > 0:
                logger.info(f"Loaded existing ChromaDB collection with {count} documents")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading existing ChromaDB collection: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar records using semantic similarity
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing search results
        """
        if self.collection is None:
            if not self.load_existing_embeddings():
                raise ValueError("No ChromaDB collection found. Please run build_embeddings_pipeline() first.")
        
        if self.model is None:
            self.load_sentence_transformer()
        
        try:
            # Search in ChromaDB collection
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Prepare results in the expected format
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    result = {
                        'rank': i + 1,
                        'score': float(similarity_score),
                        'text': doc,
                        'metadata': metadata
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise

def main():
    """Main function to execute embedding preparation"""
    # Initialize embedding preparation layer
    embedder = EmbeddingPreparationLayer()
    
    # Build embeddings pipeline
    logger.info("Starting embedding preparation pipeline...")
    result = embedder.build_embeddings_pipeline()
    
    if result['success']:
        logger.info("Embedding preparation completed successfully!")
        logger.info(f"Total embeddings: {result['total_embeddings']}")
        logger.info(f"Total documents: {result['total_documents']}")
        logger.info(f"Collection name: {result['collection_name']}")
        logger.info(f"Output directory: {result['output_directory']}")
        
        # Test search functionality
        test_queries = [
            "young male user from Lahore",
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