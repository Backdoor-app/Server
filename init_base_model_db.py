import os
import sqlite3
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration if available
try:
    import config
    DROPBOX_ENABLED = config.DROPBOX_ENABLED
    DB_PATH = config.DB_PATH
except ImportError:
    DROPBOX_ENABLED = False
    DB_PATH = "data/interactions.db"

# Create database directory for local storage
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Initialize database and tables
def init_db():
    # If Dropbox is enabled, initialize it
    if DROPBOX_ENABLED:
        try:
            from utils.dropbox_storage import init_dropbox_storage, get_dropbox_storage
            dropbox_storage = init_dropbox_storage(
                config.DROPBOX_API_KEY,
                config.DROPBOX_DB_FILENAME,
                config.DROPBOX_MODELS_FOLDER
            )
            logger.info(f"Initialized Dropbox storage for database initialization")
            
            # Get local DB path from Dropbox
            local_db_path = dropbox_storage.get_db_path()
        except Exception as e:
            logger.error(f"Failed to initialize Dropbox: {e}")
            logger.warning(f"Falling back to local database: {DB_PATH}")
            local_db_path = DB_PATH
    else:
        local_db_path = DB_PATH
    
    logger.info(f"Initializing database at {local_db_path}")
    conn = sqlite3.connect(local_db_path)
    cursor = conn.cursor()
    
    # Create model_versions table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_versions (
            version TEXT PRIMARY KEY,
            path TEXT,
            accuracy REAL,
            training_data_size INTEGER,
            training_date TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if base model entry exists
    cursor.execute("SELECT COUNT(*) FROM model_versions WHERE version = '1.0.0'")
    if cursor.fetchone()[0] == 0:
        # Add entry for our base model
        base_model_path = f"dropbox:/{config.DROPBOX_MODELS_FOLDER}/{config.BASE_MODEL_NAME}" if DROPBOX_ENABLED else f"models/{config.BASE_MODEL_NAME}"
        
        cursor.execute('''
            INSERT INTO model_versions 
            (version, path, accuracy, training_data_size, training_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            '1.0.0',  # version
            base_model_path,  # path
            0.92,  # accuracy
            1000,  # training_data_size
            datetime.now().isoformat()  # training_date
        ))
        logger.info(f"Added base model reference to database: {base_model_path}")
    else:
        logger.info("Base model already exists in database")
    
    conn.commit()
    conn.close()
    
    # Upload DB to Dropbox if using it
    if DROPBOX_ENABLED:
        try:
            dropbox_storage.upload_db()
            logger.info("Uploaded initialized database to Dropbox")
        except Exception as e:
            logger.error(f"Failed to upload database to Dropbox: {e}")

if __name__ == "__main__":
    init_db()
