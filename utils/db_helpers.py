"""
Database helper functions for the Backdoor AI learning system.

This module provides functions for:
- Initializing the database schema
- Storing and retrieving interaction data
- Managing model metadata
- Tracking model incorporation status
- Dropbox integration for database storage
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
    DROPBOX_ENABLED = getattr(config, 'DROPBOX_ENABLED', False)
except ImportError:
    DB_TIMEOUT = 60  # Default timeout if config unavailable
    DROPBOX_ENABLED = False

logger = logging.getLogger(__name__)

# Import Dropbox storage if enabled
_dropbox_storage = None
if DROPBOX_ENABLED:  
    try:
        from utils.dropbox_storage import get_dropbox_storage
        _dropbox_storage = get_dropbox_storage()
        logger.info("Dropbox storage integration enabled")
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Could not initialize Dropbox storage: {e}")
        logger.warning("Falling back to local storage")
        DROPBOX_ENABLED = False

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
    # Use Dropbox storage if enabled
    if DROPBOX_ENABLED and _dropbox_storage:
        # Get local path from Dropbox storage
        try:
            local_db_path = _dropbox_storage.get_db_path()
        except Exception as e:
            logger.error(f"Failed to get database from Dropbox: {e}")
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
            
            # Upload to Dropbox if used
            if DROPBOX_ENABLED and _dropbox_storage:
                try:
                    _dropbox_storage.upload_db()
                except Exception as e:
                    logger.error(f"Failed to upload database to Dropbox: {e}")
            
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
        
        storage_type = "Dropbox" if DROPBOX_ENABLED else "local file"
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
    
    # Upload to Dropbox if enabled
    dropbox_metadata = None
    if DROPBOX_ENABLED and _dropbox_storage:
        try:
            model_name = f"model_upload_{device_id}_{model_id}.mlmodel"
            
            # If file_path is a local file, read and upload
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    model_data = f.read()
                dropbox_metadata = _dropbox_storage.upload_model(model_data, model_name)
            # If file_path is already a path reference
            elif file_path.startswith('dropbox:') or file_path.startswith('memory:'):
                # Already uploaded to Dropbox or in memory, no need to upload again
                parts = file_path.split(':')
                if len(parts) >= 2:
                    dropbox_metadata = {'success': True, 'path': parts[1]}
            
            if dropbox_metadata and dropbox_metadata.get('success'):
                # Update file_path to include Dropbox path for reference
                file_path = f"dropbox:{dropbox_metadata['path']}"
                logger.info(f"Uploaded model to Dropbox: {dropbox_metadata['path']}")
        except Exception as e:
            logger.error(f"Failed to upload model to Dropbox: {e}")
            # Continue with memory reference
    
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
            
            # If using Dropbox, resolve file paths as needed
            if DROPBOX_ENABLED and _dropbox_storage:
                for model in models:
                    if model['file_path'].startswith('dropbox:'):
                        try:
                            # Extract model name from path
                            parts = model['file_path'].split(':')
                            if len(parts) >= 2:
                                dropbox_path = parts[1]
                                model_name = os.path.basename(dropbox_path)
                                
                                # Get streaming info from Dropbox
                                model_info = _dropbox_storage.get_model_stream(model_name)
                                if model_info and model_info.get('success'):
                                    # Update with download URL
                                    model['download_url'] = model_info.get('download_url')
                                    
                                    # For backward compatibility, also download to memory if needed
                                    memory_info = _dropbox_storage.download_model_to_memory(model_name)
                                    if memory_info and memory_info.get('success'):
                                        model['model_buffer'] = memory_info.get('model_buffer')
                        except Exception as e:
                            logger.error(f"Failed to resolve Dropbox model file: {e}")
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
            stats['storage_type'] = "dropbox" if DROPBOX_ENABLED else "local"
            
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
    # Upload to Dropbox if enabled
    dropbox_path = None
    if DROPBOX_ENABLED and _dropbox_storage:
        try:
            model_name = f"model_{version}.mlmodel"
            
            # Check if path is a string (file path) or file-like object
            if isinstance(path, str) and os.path.exists(path):
                # Upload file from path
                with open(path, 'rb') as f:
                    model_data = f.read()
                dropbox_metadata = _dropbox_storage.upload_model(model_data, model_name)
            else:
                # Try to upload directly (might be a file-like object)
                dropbox_metadata = _dropbox_storage.upload_model(path, model_name)
                
            if dropbox_metadata and dropbox_metadata.get('success'):
                dropbox_path = f"dropbox:{dropbox_metadata['path']}"
                logger.info(f"Uploaded model version {version} to Dropbox: {dropbox_metadata['path']}")
        except Exception as e:
            logger.error(f"Failed to upload model to Dropbox: {e}")
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
                dropbox_path or path,  # Use Dropbox path if available
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
    Get the path to a model file, resolving Dropbox paths if needed.
    
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
            
            # Handle Dropbox paths
            if DROPBOX_ENABLED and _dropbox_storage and path.startswith('dropbox:'):
                try:
                    parts = path.split(':')
                    if len(parts) >= 2:
                        dropbox_path = parts[1]
                        
                        # Get model name from path or use default format
                        if version == '1.0.0':
                            model_name = config.BASE_MODEL_NAME
                        else:
                            model_name = f"model_{version}.mlmodel"
                        
                        # Get streaming info from Dropbox
                        model_info = _dropbox_storage.get_model_stream(model_name)
                        if model_info and model_info.get('success'):
                            # Return streaming URL - stream endpoint will handle the redirect
                            return f"stream:{model_info.get('download_url')}"
                            
                        # Fallback to downloading to memory
                        memory_info = _dropbox_storage.download_model_to_memory(model_name)
                        if memory_info and memory_info.get('success'):
                            # Return memory buffer identifier - app will get it from the storage module
                            return f"memory:{model_name}"
                        else:
                            logger.error(f"Failed to download model {version} from Dropbox")
                            # Fall back to original path, might not exist locally
                except Exception as e:
                    logger.error(f"Error resolving Dropbox model path: {e}")
            
            # Return original path (either local or couldn't resolve Dropbox path)
            return path
            
        except Exception as e:
            logger.error(f"Error retrieving model path: {e}")
            return None
