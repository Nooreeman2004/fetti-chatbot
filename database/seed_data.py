import pandas as pd
from sqlalchemy import create_engine, text
import os
import sqlite3
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparationLayer:
    def __init__(self, db_path='database/transportation.db'):
        """Initialize the data preparation layer with database path"""
        self.db_path = db_path
        self.engine = None
        self.ensure_database_directory()
    
    def ensure_database_directory(self):
        """Create database directory if it doesn't exist"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logger.info(f"Created directory: {db_dir}")
    
    def create_database_connection(self):
        """Create SQLAlchemy engine for database connection"""
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
        logger.info(f"Database connection created: {self.db_path}")
        return self.engine
    
    def create_schema(self):
        """Create database schema with proper relationships"""
        schema_sql = """
        -- Drop existing tables if they exist (for clean setup)
        DROP TABLE IF EXISTS CheckedInUsers;
        DROP TABLE IF EXISTS TripData;
        DROP TABLE IF EXISTS CustomerDemographics;
        
        -- Customer Demographics Table
        CREATE TABLE CustomerDemographics (
            UserID INTEGER PRIMARY KEY,
            Age INTEGER,
            Gender TEXT,
            Name TEXT,
            CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Trip Data Table
        CREATE TABLE TripData (
            TripID INTEGER PRIMARY KEY,
            BookingUserID INTEGER,
            PickUpLatitude REAL,
            PickUpLongitude REAL,
            DropOffLatitude REAL,
            DropOffLongitude REAL,
            PickUpAddress TEXT,
            DropOffAddress TEXT,
            TripDate TEXT,
            TotalPassengers INTEGER,
            Duration REAL,
            Distance REAL,
            CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (BookingUserID) REFERENCES CustomerDemographics(UserID)
        );
        
        -- Checked In Users Table (Many-to-Many relationship)
        CREATE TABLE CheckedInUsers (
            CheckInID INTEGER PRIMARY KEY AUTOINCREMENT,
            UserID INTEGER,
            TripID INTEGER,
            CheckInTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CheckInStatus TEXT DEFAULT 'active',
            CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (UserID) REFERENCES CustomerDemographics(UserID),
            FOREIGN KEY (TripID) REFERENCES TripData(TripID),
            UNIQUE(UserID, TripID)
        );
        
        -- Create indexes for better performance
        CREATE INDEX idx_checkedin_userid ON CheckedInUsers(UserID);
        CREATE INDEX idx_checkedin_tripid ON CheckedInUsers(TripID);
        CREATE INDEX idx_tripdata_booking ON TripData(BookingUserID);
        CREATE INDEX idx_tripdata_date ON TripData(TripDate);
        """
        
        with self.engine.connect() as conn:
            # Execute each statement separately
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            for statement in statements:
                try:
                    conn.execute(text(statement))
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Statement execution warning: {e}")
        
        logger.info("Database schema created successfully")
    
    def load_csv_data(self, csv_files):
        """
        Load CSV data from provided file paths
        Expected csv_files format:
        {
            'trip_users': 'path/to/trip_users.csv',      # Trip ID, User ID
            'demographics': 'path/to/demographics.csv',   # User ID, Age, etc.
            'trip_details': 'path/to/trip_details.csv'   # Detailed trip info
        }
        """
        data = {}
        
        for key, file_path in csv_files.items():
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    data[key] = df
                    logger.info(f"Loaded {key}: {len(df)} rows from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {key} from {file_path}: {e}")
                    data[key] = pd.DataFrame()
            else:
                logger.warning(f"File not found: {file_path}")
                data[key] = pd.DataFrame()
        
        return data
    
    def process_demographics_data(self, df_demographics):
        """Process and clean demographics data"""
        if df_demographics.empty:
            logger.warning("Demographics data is empty")
            return pd.DataFrame()
        
        # Assuming the demographics CSV has columns like: UserID, Age
        # Add placeholder data for missing columns
        processed_df = df_demographics.copy()
        
        # Ensure required columns exist
        if 'UserID' not in processed_df.columns:
            if len(processed_df.columns) >= 1:
                processed_df.rename(columns={processed_df.columns[0]: 'UserID'}, inplace=True)
        
        if 'Age' not in processed_df.columns:
            if len(processed_df.columns) >= 2:
                processed_df.rename(columns={processed_df.columns[1]: 'Age'}, inplace=True)
            else:
                processed_df['Age'] = None
        
        # Add default values for missing columns
        if 'Gender' not in processed_df.columns:
            processed_df['Gender'] = 'Unknown'
        
        if 'Name' not in processed_df.columns:
            processed_df['Name'] = processed_df['UserID'].apply(lambda x: f'User_{x}')
        
        # Clean data
        processed_df['UserID'] = pd.to_numeric(processed_df['UserID'], errors='coerce')
        processed_df['Age'] = pd.to_numeric(processed_df['Age'], errors='coerce')
        processed_df = processed_df.dropna(subset=['UserID'])
        
        logger.info(f"Processed demographics data: {len(processed_df)} records")
        return processed_df
    
    def process_trip_details_data(self, df_trip_details):
        """Process and clean trip details data"""
        if df_trip_details.empty:
            logger.warning("Trip details data is empty")
            return pd.DataFrame()
        
        processed_df = df_trip_details.copy()
        
        # Map columns based on actual CSV structure
        column_mapping = {
            'Trip ID': 'TripID',
            'Booking User ID': 'BookingUserID',
            'Pick Up Latitude': 'PickUpLatitude',
            'Pick Up Longitude': 'PickUpLongitude',
            'Drop Off Latitude': 'DropOffLatitude',
            'Drop Off Longitude': 'DropOffLongitude',
            'Pick Up Address': 'PickUpAddress',
            'Drop Off Address': 'DropOffAddress',
            'Trip Date and Time': 'TripDate',
            'Total Passengers': 'TotalPassengers'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in processed_df.columns:
                processed_df.rename(columns={old_name: new_name}, inplace=True)
        
        # If columns don't match expected names, try to map by position
        if 'TripID' not in processed_df.columns and len(processed_df.columns) >= 1:
            processed_df.rename(columns={processed_df.columns[0]: 'TripID'}, inplace=True)
        
        if 'BookingUserID' not in processed_df.columns and len(processed_df.columns) >= 2:
            processed_df.rename(columns={processed_df.columns[1]: 'BookingUserID'}, inplace=True)
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['TripID', 'BookingUserID', 'PickUpLatitude', 'PickUpLongitude', 
                          'DropOffLatitude', 'DropOffLongitude', 'TotalPassengers']
        
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # Add calculated fields
        if 'Duration' not in processed_df.columns:
            processed_df['Duration'] = None
        
        if 'Distance' not in processed_df.columns:
            processed_df['Distance'] = None
        
        # Clean data
        processed_df = processed_df.dropna(subset=['TripID'])

        # Normalize TripDate to ISO format for reliable SQLite time functions
        if 'TripDate' in processed_df.columns:
            try:
                parsed_dt = pd.to_datetime(processed_df['TripDate'], errors='coerce')
            except Exception:
                parsed_dt = pd.to_datetime(processed_df['TripDate'], errors='coerce', dayfirst=False)
            processed_df['TripDate'] = parsed_dt.dt.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Processed trip details data: {len(processed_df)} records")
        return processed_df
    
    def process_trip_users_data(self, df_trip_users):
        """Process trip-user relationship data"""
        if df_trip_users.empty:
            logger.warning("Trip-users data is empty")
            return pd.DataFrame()
        
        processed_df = df_trip_users.copy()
        
        # Ensure proper column names
        if 'TripID' not in processed_df.columns and len(processed_df.columns) >= 1:
            processed_df.rename(columns={processed_df.columns[0]: 'TripID'}, inplace=True)
        
        if 'UserID' not in processed_df.columns and len(processed_df.columns) >= 2:
            processed_df.rename(columns={processed_df.columns[1]: 'UserID'}, inplace=True)
        
        # Convert to numeric
        processed_df['TripID'] = pd.to_numeric(processed_df['TripID'], errors='coerce')
        processed_df['UserID'] = pd.to_numeric(processed_df['UserID'], errors='coerce')
        
        # Remove invalid records
        processed_df = processed_df.dropna(subset=['TripID', 'UserID'])
        
        # Add check-in timestamp
        processed_df['CheckInTime'] = datetime.now().isoformat()
        processed_df['CheckInStatus'] = 'active'
        
        logger.info(f"Processed trip-users data: {len(processed_df)} records")
        return processed_df
    
    def load_data_to_database(self, data):
        """Load processed data into database tables"""
        try:
            # Load Customer Demographics
            if not data['demographics'].empty:
                demographics_df = self.process_demographics_data(data['demographics'])
                if not demographics_df.empty:
                    demographics_df.to_sql('CustomerDemographics', self.engine, 
                                         if_exists='append', index=False)
                    logger.info(f"Loaded {len(demographics_df)} customer demographics records")
            
            # Load Trip Data
            if not data['trip_details'].empty:
                trip_details_df = self.process_trip_details_data(data['trip_details'])
                if not trip_details_df.empty:
                    trip_details_df.to_sql('TripData', self.engine, 
                                         if_exists='append', index=False)
                    logger.info(f"Loaded {len(trip_details_df)} trip data records")
            
            # Load Checked In Users
            if not data['trip_users'].empty:
                trip_users_df = self.process_trip_users_data(data['trip_users'])
                if not trip_users_df.empty:
                    trip_users_df.to_sql('CheckedInUsers', self.engine, 
                                       if_exists='append', index=False)
                    logger.info(f"Loaded {len(trip_users_df)} checked-in user records")
            
            logger.info("All data loaded successfully into database")

            # Create helpful derived views for robust analytics
            with self.engine.connect() as conn:
                try:
                    # View: TripFeatures - normalized time fields and coarse location labels
                    conn.execute(text("""
                        DROP VIEW IF EXISTS TripFeatures;
                        CREATE VIEW TripFeatures AS
                        SELECT 
                            t.TripID,
                            t.BookingUserID,
                            t.PickUpAddress,
                            t.DropOffAddress,
                            t.TotalPassengers,
                            t.TripDate AS TripDateTime,
                            DATE(t.TripDate) AS TripDate,
                            STRFTIME('%Y-%m', t.TripDate) AS TripYearMonth,
                            CAST(STRFTIME('%H', t.TripDate) AS INTEGER) AS TripHour,
                            CAST(STRFTIME('%w', t.TripDate) AS INTEGER) AS TripWeekday,
                            CASE WHEN STRFTIME('%w', t.TripDate) IN ('0','6') THEN 1 ELSE 0 END AS IsWeekend,
                            -- Coarse pickup label
                            CASE 
                                WHEN t.PickUpAddress LIKE '%West Campus%' THEN 'West Campus'
                                WHEN t.PickUpAddress LIKE '%South Congress%' THEN 'South Congress'
                                WHEN t.PickUpAddress LIKE '%The Drag%' THEN 'The Drag'
                                WHEN t.PickUpAddress LIKE '%Downtown%' OR t.PickUpAddress LIKE '%6th%' THEN 'Downtown'
                                ELSE 'Other'
                            END AS PickUpLabel,
                            -- Coarse dropoff label
                            CASE 
                                WHEN t.DropOffAddress LIKE '%Moody%' THEN 'Moody Center'
                                WHEN t.DropOffAddress LIKE '%Airport%' OR t.DropOffAddress LIKE '%AUS%' THEN 'Airport'
                                WHEN t.DropOffAddress LIKE '%West Campus%' THEN 'West Campus'
                                WHEN t.DropOffAddress LIKE '%South Congress%' THEN 'South Congress'
                                WHEN t.DropOffAddress LIKE '%The Drag%' THEN 'The Drag'
                                WHEN t.DropOffAddress LIKE '%Downtown%' OR t.DropOffAddress LIKE '%6th%' THEN 'Downtown'
                                ELSE 'Other'
                            END AS DropOffLabel
                        FROM TripData t;
                    """))

                    # View: TripPassengerAges - per trip passenger ages
                    conn.execute(text("""
                        DROP VIEW IF EXISTS TripPassengerAges;
                        CREATE VIEW TripPassengerAges AS
                        SELECT 
                            ci.TripID,
                            cd.UserID,
                            cd.Age
                        FROM CheckedInUsers ci
                        JOIN CustomerDemographics cd ON cd.UserID = ci.UserID
                        WHERE cd.Age IS NOT NULL;
                    """))

                    # View: TripPassengerStats - per trip aggregated age stats
                    conn.execute(text("""
                        DROP VIEW IF EXISTS TripPassengerStats;
                        CREATE VIEW TripPassengerStats AS
                        SELECT 
                            t.TripID,
                            COUNT(*) AS passenger_count,
                            AVG(Age) AS avg_age,
                            MIN(Age) AS min_age,
                            MAX(Age) AS max_age,
                            SUM(CASE WHEN Age < 18 THEN 1 ELSE 0 END) AS under18_count,
                            SUM(CASE WHEN Age BETWEEN 18 AND 24 THEN 1 ELSE 0 END) AS age18_24_count,
                            SUM(CASE WHEN Age BETWEEN 20 AND 25 THEN 1 ELSE 0 END) AS age20_25_count,
                            SUM(CASE WHEN Age > 30 THEN 1 ELSE 0 END) AS over30_count
                        FROM TripPassengerAges t
                        GROUP BY t.TripID;
                    """))
                    conn.commit()
                    logger.info("Derived views created: TripFeatures, TripPassengerAges, TripPassengerStats")
                except Exception as e:
                    logger.warning(f"Creating derived views warning: {e}")
            
        except Exception as e:
            logger.error(f"Error loading data to database: {e}")
            raise
    
    def validate_data_integrity(self):
        """Validate foreign key relationships and data integrity"""
        try:
            with self.engine.connect() as conn:
                # Check for orphaned records
                orphaned_trips = conn.execute(text("""
                    SELECT COUNT(*) as count FROM TripData t 
                    LEFT JOIN CustomerDemographics c ON t.BookingUserID = c.UserID 
                    WHERE c.UserID IS NULL AND t.BookingUserID IS NOT NULL
                """)).fetchone()
                
                orphaned_checkins = conn.execute(text("""
                    SELECT COUNT(*) as count FROM CheckedInUsers ci 
                    LEFT JOIN CustomerDemographics c ON ci.UserID = c.UserID 
                    WHERE c.UserID IS NULL
                """)).fetchone()
                
                orphaned_checkins_trips = conn.execute(text("""
                    SELECT COUNT(*) as count FROM CheckedInUsers ci 
                    LEFT JOIN TripData t ON ci.TripID = t.TripID 
                    WHERE t.TripID IS NULL
                """)).fetchone()
                
                # Get record counts
                demographics_count = conn.execute(text("SELECT COUNT(*) FROM CustomerDemographics")).fetchone()[0]
                trips_count = conn.execute(text("SELECT COUNT(*) FROM TripData")).fetchone()[0]
                checkins_count = conn.execute(text("SELECT COUNT(*) FROM CheckedInUsers")).fetchone()[0]
                
                logger.info(f"Data validation results:")
                logger.info(f"  Customer Demographics: {demographics_count} records")
                logger.info(f"  Trip Data: {trips_count} records")
                logger.info(f"  Checked In Users: {checkins_count} records")
                logger.info(f"  Orphaned trips (no user): {orphaned_trips[0]}")
                logger.info(f"  Orphaned check-ins (no user): {orphaned_checkins[0]}")
                logger.info(f"  Orphaned check-ins (no trip): {orphaned_checkins_trips[0]}")
                
                return {
                    'demographics_count': demographics_count,
                    'trips_count': trips_count,
                    'checkins_count': checkins_count,
                    'orphaned_trips': orphaned_trips[0],
                    'orphaned_checkins': orphaned_checkins[0],
                    'orphaned_checkins_trips': orphaned_checkins_trips[0]
                }
        
        except Exception as e:
            logger.error(f"Error validating data integrity: {e}")
            return None

def main():
    """Main function to execute data preparation pipeline"""
    # Initialize data preparation layer
    prep = DataPreparationLayer()
    
    # Create database connection
    prep.create_database_connection()
    
    # Create schema
    prep.create_schema()
    
    # Define CSV file paths (update these paths as needed)
    csv_files = {
        'trip_users': 'data/CheckedIn_UserID\'s.csv',        # Trip ID, User ID relationship
        'demographics': 'data/CustomerDemographics.csv',     # User demographics
        'trip_details': 'data/TripData.csv'     # Detailed trip information
    }
    
    # Load CSV data
    data = prep.load_csv_data(csv_files)
    
    # Load data into database
    if any(not df.empty for df in data.values()):
        prep.load_data_to_database(data)
        
        # Validate data integrity
        validation_results = prep.validate_data_integrity()
        
        if validation_results:
            logger.info("Data preparation completed successfully!")
        else:
            logger.warning("Data preparation completed with validation issues")
    else:
        logger.warning("No data found to load. Please check your CSV file paths.")

if __name__ == "__main__":
    main()