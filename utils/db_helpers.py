"""
Database helper functions for the Backdoor AI learning system.

This module provides functions for:
- Initializing the database schema
- Storing and retrieving interaction data
- Managing model metadata
- Tracking model incorporation status
- Google Drive integration for database storage
"""

import sqlite3
import os
import logging
import uuid
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from contextlib import contextmanager

# Import configuration for timeout values
try:
    import config
    DB_TIMEOUT = getattr(config, 'DB_LOCK_TIMEOUT', 60)
    GOOGLE_DRIVE_ENABLED = getattr(config, 'GOOGLE_DRIVE_ENABLED', False)
except ImportError:
    DB_TIMEOUT = 60  # Default timeout if config unavailable
    GOOGLE_DRIVE_ENABLED = False

logger = logging.getLogger(__name__)

# Import Google Drive storage if enabled
_drive_storage = None
if GOOGLE_DRIVE_ENABLED:
    try:
        from utils.drive_storage import get_drive_storage
        _drive_storage = get_drive_storage()
        logger.info("Google Drive storage integration enabled")
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Could not initialize Google Drive storage: {e}")
        logger.warning("Falling back to local storage")
        GOOGLE_DRIVE_ENABLED = False

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
    # Use Google Drive storage if enabled
    if GOOGLE_DRIVE_ENABLED and _drive_storage:
        # Get local path from Drive storage
        try:
            local_db_path = _drive_storage.get_db_path()
        except Exception as e:
            logger.error(f"Failed to get database from Google Drive: {e}")
            logger.warning(f"Falling back to local database at {db_path}")
            local_db_path = db_path
    else:
        local_db_path = db_path
        
    conn = None
    retries = 3  # Retry connections in case of database lock
    last_error = None
    
    while retries > 0:
        try:
            conn = sqlite3.connect(local_db_path, timeout=DB_TIMEOUT)
            if row_factory:
                conn.row_factory = sqlite3.Row
            yield conn
            
            # Upload to Google Drive if used
            if GOOGLE_DRIVE_ENABLED and _drive_storage:
                try:
                    _drive_storage.upload_db()
                except Exception as e:
                    logger.error(f"Failed to upload database to Google Drive: {e}")
            
            # Break retry loop on success
            break
            
        except sqlite3.OperationalError as e:
            # Retry if database is locked
            if "database is locked" in str(e) and retries > 1:
                retries -= 1
                last_error = e
                # Random backoff to reduce contention
                time.sleep(random.uniform(0.5, 2.0))
            else:
                logger.error(f"Database operational error: {e}")
                if conn:
                    conn.rollback()
                raise
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    # If we've exhausted retries, raise the last error
    if retries == 0 and last_error:
        raise last_error

def init_db(db_path: str) -> None:
    """
    Initialize the database schema if tables don't exist.
    
    Args:
        db_path: Path to the SQLite database
    """
    # Ensure directory exists for local storage
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
        
        storage_type = "Google Drive" if GOOGLE_DRIVE_ENABLED else "local file"
        logger.info(f"Database initialized using {storage_type} storage at {db_path}")

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
    
    # Upload to Google Drive if enabled
    drive_metadata = None
    if GOOGLE_DRIVE_ENABLED and _drive_storage and os.path.exists(file_path):
        try:
            model_name = f"model_upload_{device_id}_{model_id}.mlmodel"
            drive_metadata = _drive_storage.upload_model(file_path, model_name)
            if drive_metadata and drive_metadata.get('success'):
                # Update file_path to include Google Drive ID for reference
                file_path = f"gdrive:{drive_metadata['id']}:{file_path}"
                logger.info(f"Uploaded model to Google Drive: {drive_metadata['id']}")
        except Exception as e:
            logger.error(f"Failed to upload model to Google Drive: {e}")
            # Continue with local reference only
    
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
            
            models = [dict(row) for row in cursor.fetchall()]
            
            # If using Google Drive, resolve file paths as needed
            if GOOGLE_DRIVE_ENABLED and _drive_storage:
                for model in models:
                    if model['file_path'].startswith('gdrive:'):
                        try:
                            # Extract model name from path
                            parts = model['file_path'].split(':')
                            if len(parts) >= 3:
                                drive_id = parts[1]
                                original_path = ':'.join(parts[2:])
                                model_name = os.path.basename(original_path)
                                
                                # Download from Google Drive
                                download_info = _drive_storage.download_model(model_name)
                                if download_info and download_info.get('success'):
                                    # Update file_path to local path
                                    model['file_path'] = download_info['local_path']
                        except Exception as e:
                            logger.error(f"Failed to resolve Google Drive model file: {e}")
                            # Keep original path, will need to be handled downstream
            
            return models
            
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
                
            # Add storage type information
            stats['storage_type'] = "google_drive" if GOOGLE_DRIVE_ENABLED else "local"
            
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
    # Upload to Google Drive if enabled
    drive_path = None
    if GOOGLE_DRIVE_ENABLED and _drive_storage and os.path.exists(path):
        try:
            model_name = f"model_{version}.mlmodel"
            drive_metadata = _drive_storage.upload_model(path, model_name)
            if drive_metadata and drive_metadata.get('success'):
                drive_path = f"gdrive:{drive_metadata['id']}:{path}"
                logger.info(f"Uploaded model version {version} to Google Drive: {drive_metadata['id']}")
        except Exception as e:
            logger.error(f"Failed to upload model to Google Drive: {e}")
            # Continue with local storage only
    
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
                drive_path or path,  # Use Drive path if available
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

def get_model_path(db_path: str, version: str) -> Optional[str]:
    """
    Get the path to a model file, resolving Google Drive paths if needed.
    
    Args:
        db_path: Path to the SQLite database
        version: Version of the model to retrieve
        
    Returns:
        Optional[str]: Local path to the model file or None if not found
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT path FROM model_versions WHERE version = ?", (version,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Model version {version} not found in database")
                return None
                
            path = result[0]
            
            # Handle Google Drive paths
            if GOOGLE_DRIVE_ENABLED and _drive_storage and path.startswith('gdrive:'):
                try:
                    parts = path.split(':')
                    if len(parts) >= 3:
                        drive_id = parts[1]
                        original_path = ':'.join(parts[2:])
                        model_name = f"model_{version}.mlmodel"
                        
                        # Download from Google Drive
                        download_info = _drive_storage.download_model(model_name)
                        if download_info and download_info.get('success'):
                            return download_info['local_path']
                        else:
                            logger.error(f"Failed to download model {version} from Google Drive")
                            # Fall back to original path, might not exist locally
                except Exception as e:
                    logger.error(f"Error resolving Google Drive model path: {e}")
            
            # Return original path (either local or couldn't resolve Drive path)
            return path
            
        except Exception as e:
            logger.error(f"Error retrieving model path: {e}")
            return None
