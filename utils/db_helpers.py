import sqlite3
import os
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id TEXT PRIMARY KEY,
            device_id TEXT,
            timestamp TEXT,
            user_message TEXT,
            ai_response TEXT,
            detected_intent TEXT,
            confidence_score REAL,
            app_version TEXT,
            model_version TEXT,
            os_version TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            interaction_id TEXT PRIMARY KEY,
            rating INTEGER,
            comment TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (interaction_id) REFERENCES interactions (id)
        )
    ''')
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
    # New table for uploaded models
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_models (
            id TEXT PRIMARY KEY,
            device_id TEXT,
            app_version TEXT,
            description TEXT,
            file_path TEXT,
            file_size INTEGER,
            original_filename TEXT,
            upload_date TEXT,
            incorporated_in_version TEXT,
            incorporation_status TEXT DEFAULT 'pending', -- pending, processing, incorporated, failed
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Table for tracking ensemble models
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ensemble_models (
            ensemble_version TEXT PRIMARY KEY,
            description TEXT,
            component_models TEXT, -- JSON array of model IDs that make up this ensemble
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")

def store_interactions(db_path, data):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        conn.execute("BEGIN TRANSACTION")
        for interaction in data['interactions']:
            cursor.execute('''
                INSERT OR REPLACE INTO interactions 
                (id, device_id, timestamp, user_message, ai_response, detected_intent, confidence_score, app_version, model_version, os_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                interaction['id'],
                data['deviceId'],
                interaction['timestamp'],
                interaction['userMessage'],
                interaction['aiResponse'],
                interaction['detectedIntent'],
                interaction['confidenceScore'],
                data['appVersion'],
                data['modelVersion'],
                data['osVersion']
            ))
            if 'feedback' in interaction and interaction['feedback']:
                cursor.execute('''
                    INSERT OR REPLACE INTO feedback 
                    (interaction_id, rating, comment)
                    VALUES (?, ?, ?)
                ''', (
                    interaction['id'],
                    interaction['feedback']['rating'],
                    interaction['feedback'].get('comment')
                ))
        conn.commit()
        logger.info(f"Stored {len(data['interactions'])} interactions from device {data['deviceId']}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing interactions: {str(e)}")
        raise
    finally:
        conn.close()

def store_uploaded_model(db_path, device_id, app_version, description, file_path, file_size, original_filename):
    """
    Store metadata about an uploaded model in the database
    """
    model_id = str(uuid.uuid4())
    upload_date = datetime.now().isoformat()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO uploaded_models
            (id, device_id, app_version, description, file_path, file_size, original_filename, upload_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            device_id,
            app_version,
            description,
            file_path,
            file_size,
            original_filename,
            upload_date
        ))
        conn.commit()
        logger.info(f"Stored metadata for uploaded model: {model_id} from device {device_id}")
        return model_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing uploaded model metadata: {str(e)}")
        raise
    finally:
        conn.close()

def update_model_incorporation_status(db_path, model_id, status, version=None):
    """
    Update the status of an uploaded model's incorporation into the ensemble
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        if version:
            cursor.execute('''
                UPDATE uploaded_models
                SET incorporation_status = ?, incorporated_in_version = ?
                WHERE id = ?
            ''', (status, version, model_id))
        else:
            cursor.execute('''
                UPDATE uploaded_models
                SET incorporation_status = ?
                WHERE id = ?
            ''', (status, model_id))
        conn.commit()
        logger.info(f"Updated incorporation status for model {model_id} to {status}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating model incorporation status: {str(e)}")
        raise
    finally:
        conn.close()

def get_pending_uploaded_models(db_path):
    """
    Get all uploaded models that haven't been incorporated into an ensemble yet
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT * FROM uploaded_models
            WHERE incorporation_status IN ('pending', 'processing')
        ''')
        return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error retrieving pending uploaded models: {str(e)}")
        return []
    finally:
        conn.close()
