"""
Database helper functions for the Backdoor AI learning system.

This module provides functions for:
- Initializing the database schema
- Storing and retrieving interaction data
- Managing model metadata
- Tracking model incorporation status
"""

import sqlite3
import os
import logging
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from contextlib import contextmanager

# Import configuration for timeout values
try:
    import config
    DB_TIMEOUT = getattr(config, 'DB_LOCK_TIMEOUT', 60)
except ImportError:
    DB_TIMEOUT = 60  # Default timeout if config unavailable

logger = logging.getLogger(__name__)

@contextmanager
def get_connection(db_path: str, row_factory: bool = False):
    """
    Context manager for database connections.
    
    Args:
        db_path: Path to the SQLite database
        row_factory: Whether to use sqlite3.Row as row factory
        
    Yields:
        sqlite3.Connection: Database connection
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=DB_TIMEOUT)
        if row_factory:
            conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def init_db(db_path: str) -> None:
    """
    Initialize the database schema if tables don't exist.
    
    Args:
        db_path: Path to the SQLite database
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Interactions table
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
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                interaction_id TEXT PRIMARY KEY,
                rating INTEGER,
                comment TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (interaction_id) REFERENCES interactions (id)
            )
        ''')
        
        # Model versions table
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
        
        # Uploaded models table
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
        
        # Ensemble models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ensemble_models (
                ensemble_version TEXT PRIMARY KEY,
                description TEXT,
                component_models TEXT, -- JSON array of model IDs that make up this ensemble
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add index for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_device ON interactions(device_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_intent ON interactions(detected_intent)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_uploaded_status ON uploaded_models(incorporation_status)')
        
        conn.commit()
        logger.info(f"Database initialized at {db_path}")

def store_interactions(db_path: str, data: Dict[str, Any]) -> int:
    """
    Store interaction data from devices.
    
    Args:
        db_path: Path to the SQLite database
        data: Dictionary containing device info and interactions
        
    Returns:
        int: Number of interactions stored
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        try:
            conn.execute("BEGIN TRANSACTION")
            interaction_count = 0
            
            for interaction in data.get('interactions', []):
                cursor.execute('''
                    INSERT OR REPLACE INTO interactions 
                    (id, device_id, timestamp, user_message, ai_response, detected_intent, confidence_score, app_version, model_version, os_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    interaction.get('id'),
                    data.get('deviceId', 'unknown'),
                    interaction.get('timestamp'),
                    interaction.get('userMessage'),
                    interaction.get('aiResponse'),
                    interaction.get('detectedIntent'),
                    interaction.get('confidenceScore', 0.0),
                    data.get('appVersion'),
                    data.get('modelVersion'),
                    data.get('osVersion')
                ))
                interaction_count += 1
                
                # Store feedback if available
                if 'feedback' in interaction and interaction['feedback']:
                    cursor.execute('''
                        INSERT OR REPLACE INTO feedback 
                        (interaction_id, rating, comment)
                        VALUES (?, ?, ?)
                    ''', (
                        interaction.get('id'),
                        interaction['feedback'].get('rating'),
                        interaction['feedback'].get('comment')
                    ))
                    
            conn.commit()
            logger.info(f"Stored {interaction_count} interactions from device {data.get('deviceId', 'unknown')}")
            return interaction_count
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing interactions: {e}")
            raise

def store_uploaded_model(
    db_path: str,
    device_id: str,
    app_version: str,
    description: str,
    file_path: str,
    file_size: int,
    original_filename: str
) -> str:
    """
    Store metadata about an uploaded model in the database.
    
    Args:
        db_path: Path to the SQLite database
        device_id: ID of the device that uploaded the model
        app_version: Version of the app used to create the model
        description: User-provided description of the model
        file_path: Path where the model file is stored
        file_size: Size of the model file in bytes
        original_filename: Original filename of the uploaded model
        
    Returns:
        str: Generated UUID for the model entry
    """
    model_id = str(uuid.uuid4())
    upload_date = datetime.now().isoformat()
    
    with get_connection(db_path) as conn:
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
            logger.error(f"Error storing uploaded model metadata: {e}")
            raise

def update_model_incorporation_status(
    db_path: str,
    model_id: str,
    status: str,
    version: Optional[str] = None
) -> bool:
    """
    Update the status of an uploaded model's incorporation into the ensemble.
    
    Args:
        db_path: Path to the SQLite database
        model_id: ID of the model to update
        status: New status (pending, processing, incorporated, failed)
        version: Version of the ensemble model it was incorporated into (optional)
        
    Returns:
        bool: True if update was successful
    """
    with get_connection(db_path) as conn:
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
                
            rows_affected = cursor.rowcount
            conn.commit()
            
            logger.info(f"Updated incorporation status for model {model_id} to {status}")
            return rows_affected > 0
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating model incorporation status: {e}")
            raise

def get_pending_uploaded_models(db_path: str) -> List[Dict[str, Any]]:
    """
    Get all uploaded models that haven't been incorporated into an ensemble yet.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        List of dictionaries containing model information
    """
    with get_connection(db_path, row_factory=True) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT * FROM uploaded_models
                WHERE incorporation_status IN ('pending', 'processing')
                ORDER BY upload_date ASC
            ''')
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error retrieving pending uploaded models: {e}")
            return []

def get_model_stats(db_path: str) -> Dict[str, Any]:
    """
    Get statistics about models and training data.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Dictionary with statistics
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        try:
            stats = {}
            
            # Get total models trained
            cursor.execute("SELECT COUNT(*) FROM model_versions")
            stats['total_models'] = cursor.fetchone()[0]
            
            # Get incorporated user models
            cursor.execute("SELECT COUNT(*) FROM uploaded_models WHERE incorporation_status = 'incorporated'")
            stats['incorporated_models'] = cursor.fetchone()[0]
            
            # Get pending user models
            cursor.execute("SELECT COUNT(*) FROM uploaded_models WHERE incorporation_status = 'pending'")
            stats['pending_models'] = cursor.fetchone()[0]
            
            # Get failed incorporations
            cursor.execute("SELECT COUNT(*) FROM uploaded_models WHERE incorporation_status = 'failed'")
            stats['failed_incorporations'] = cursor.fetchone()[0]
            
            # Get latest model details
            cursor.execute("""
                SELECT version, accuracy, training_data_size, training_date 
                FROM model_versions 
                ORDER BY created_at DESC LIMIT 1
            """)
            latest = cursor.fetchone()
            if latest:
                stats['latest_version'] = latest[0]
                stats['latest_accuracy'] = latest[1]
                stats['latest_training_size'] = latest[2]
                stats['latest_training_date'] = latest[3]
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting model stats: {e}")
            return {"error": str(e)}

def store_model_version(
    db_path: str, 
    version: str, 
    path: str, 
    accuracy: float, 
    training_data_size: int, 
    is_ensemble: bool = False,
    component_models: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    Store information about a newly trained model.
    
    Args:
        db_path: Path to the SQLite database
        version: Version string of the model
        path: File path to the model
        accuracy: Model accuracy from validation
        training_data_size: Number of samples used for training
        is_ensemble: Whether this is an ensemble model
        component_models: List of component model info for ensemble models
        
    Returns:
        bool: Whether the operation was successful
    """
    training_date = datetime.now().isoformat()
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        try:
            # Insert model version
            cursor.execute('''
                INSERT INTO model_versions 
                (version, path, accuracy, training_data_size, training_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                version,
                path,
                float(accuracy),
                training_data_size,
                training_date
            ))
            
            # Store ensemble info if applicable
            if is_ensemble and component_models:
                component_json = json.dumps(component_models)
                description = f"Ensemble model with {len(component_models)} component models"
                
                cursor.execute('''
                    INSERT INTO ensemble_models 
                    (ensemble_version, description, component_models)
                    VALUES (?, ?, ?)
                ''', (
                    version,
                    description,
                    component_json
                ))
                
            conn.commit()
            logger.info(f"Stored new model version: {version}")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing model version: {e}")
            return False
