import sqlite3
import os
import logging
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
