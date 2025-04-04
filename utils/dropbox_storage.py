"""
Dropbox integration for Backdoor AI server.

This module provides functionality to:
- Store and retrieve SQLite database files on Dropbox
- Upload and download ML model files
- Handle authentication and synchronization
"""

import os
import tempfile
import time
import threading
import logging
import json
import dropbox
from dropbox.exceptions import ApiError, AuthError
from dropbox.files import WriteMode
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class DropboxStorage:
    """Handles Dropbox storage operations for the Backdoor AI server."""
    
    def __init__(self, api_key: str, db_filename: str = "interactions.db", models_folder_name: str = "backdoor_models"):
        """
        Initialize Dropbox storage.
        
        Args:
            api_key: Dropbox API key
            db_filename: Name of the database file in Dropbox
            models_folder_name: Name of the folder to store models in
        """
        self.api_key = api_key
        self.db_filename = db_filename
        self.models_folder_name = models_folder_name
        
        # File tracking
        self.model_files = {}  # Mapping of model names to paths
        
        # Local storage paths
        self.temp_dir = tempfile.gettempdir()
        self.local_db_path = os.path.join(self.temp_dir, db_filename)
        
        # Sync state
        self.last_db_sync = 0
        self.last_models_sync = 0
        self.db_sync_interval = 60  # Seconds
        self.models_sync_interval = 300  # Seconds
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Connect to Dropbox
        self.dbx = self._authenticate()
        
        # Initialize resources
        self._initialize()
        
    def _authenticate(self) -> dropbox.Dropbox:
        """
        Authenticate with Dropbox using the provided API key.
        
        Returns:
            dropbox.Dropbox: Authenticated Dropbox instance
        
        Raises:
            Exception: If authentication fails
        """
        try:
            # Initialize Dropbox client
            dbx = dropbox.Dropbox(self.api_key)
            # Check if the access token is valid
            dbx.users_get_current_account()
            logger.info("Successfully authenticated with Dropbox")
            return dbx
            
        except AuthError as e:
            logger.error(f"Dropbox authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Dropbox initialization error: {e}")
            raise
            
    def _initialize(self) -> None:
        """Initialize Dropbox resources (find or create DB file and models folder)."""
        try:
            # Check if models folder exists, create if it doesn't
            self._find_or_create_models_folder()
            
            # Download database if it exists
            self._download_db()
                
            # Sync model file list
            self._sync_model_files()
            
            logger.info("Dropbox storage initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Dropbox storage: {e}")
            raise
    
    def _download_db(self) -> bool:
        """
        Download the database from Dropbox.
        
        Returns:
            bool: True if download was successful, False otherwise
        """
        db_path = f"/{self.db_filename}"
        try:
            # Check if file exists on Dropbox
            try:
                self.dbx.files_get_metadata(db_path)
                file_exists = True
            except ApiError as e:
                if e.error.is_path() and e.error.get_path().is_not_found():
                    file_exists = False
                else:
                    raise
            
            if file_exists:
                # Download file
                self.dbx.files_download_to_file(self.local_db_path, db_path)
                self.last_db_sync = time.time()
                
                logger.info(f"Downloaded database from Dropbox: {self.local_db_path}")
                return True
            else:
                logger.info(f"Database file '{self.db_filename}' not found in Dropbox, will create on first upload")
                return False
            
        except Exception as e:
            logger.error(f"Error downloading database: {e}")
            return False
    
    def _find_or_create_models_folder(self) -> bool:
        """
        Find or create the models folder in Dropbox.
        
        Returns:
            bool: True if folder was found or created, False otherwise
        """
        folder_path = f"/{self.models_folder_name}"
        try:
            # Check if folder exists
            try:
                self.dbx.files_get_metadata(folder_path)
                logger.info(f"Found models folder in Dropbox: {folder_path}")
                return True
            except ApiError as e:
                if e.error.is_path() and e.error.get_path().is_not_found():
                    # Create folder if it doesn't exist
                    self.dbx.files_create_folder_v2(folder_path)
                    logger.info(f"Created models folder in Dropbox: {folder_path}")
                    return True
                else:
                    raise
            
        except Exception as e:
            logger.error(f"Error finding or creating models folder: {e}")
            return False
    
    def _sync_model_files(self) -> bool:
        """
        Sync the list of model files from Dropbox.
        
        Returns:
            bool: True if sync was successful, False otherwise
        """
        folder_path = f"/{self.models_folder_name}"
        try:
            try:
                # List all files in the models folder
                result = self.dbx.files_list_folder(folder_path)
                
                # Update model files map
                self.model_files = {}
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith('.mlmodel'):
                        self.model_files[entry.name] = entry.path_display
                
                # Continue listing if there are more files
                while result.has_more:
                    result = self.dbx.files_list_folder_continue(result.cursor)
                    for entry in result.entries:
                        if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith('.mlmodel'):
                            self.model_files[entry.name] = entry.path_display
                
                self.last_models_sync = time.time()
                
                logger.info(f"Synced {len(self.model_files)} model files from Dropbox")
                return True
            
            except ApiError as e:
                if e.error.is_path() and e.error.get_path().is_not_found():
                    # Create the folder if it doesn't exist
                    self._find_or_create_models_folder()
                    # Models folder was empty or didn't exist
                    self.model_files = {}
                    self.last_models_sync = time.time()
                    return True
                else:
                    raise
                
        except Exception as e:
            logger.error(f"Error syncing model files: {e}")
            return False
    
    def get_db_path(self) -> str:
        """
        Get the local path to the database file, downloading it from Dropbox if needed.
        
        Returns:
            str: Path to the local database file
        """
        with self.lock:
            # Check if we need to sync from Dropbox
            current_time = time.time()
            if current_time - self.last_db_sync > self.db_sync_interval:
                self._download_db()
            
            return self.local_db_path
    
    def upload_db(self) -> bool:
        """
        Upload the database to Dropbox.
        
        Returns:
            bool: True if upload was successful, False otherwise
        """
        with self.lock:
            if not os.path.exists(self.local_db_path):
                logger.warning(f"Cannot upload database: File not found at {self.local_db_path}")
                return False
                
            try:
                # Upload file with overwrite mode
                with open(self.local_db_path, 'rb') as f:
                    self.dbx.files_upload(f.read(), f"/{self.db_filename}", 
                                         mode=WriteMode.overwrite)
                
                self.last_db_sync = time.time()
                
                logger.info(f"Uploaded database to Dropbox: /{self.db_filename}")
                return True
                
            except Exception as e:
                logger.error(f"Error uploading database: {e}")
                return False
    
    def upload_model(self, data_or_path, model_name: str) -> Dict[str, Any]:
        """
        Upload a model file to Dropbox.
        
        Args:
            data_or_path: Either a file path, binary data, or file-like object
            model_name: Name to use for the model in Dropbox
            
        Returns:
            Dict with model information (success, name, path, etc.)
        """
        with self.lock:
            dropbox_path = f"/{self.models_folder_name}/{model_name}"
            file_data = None
            file_size = 0
            
            try:
                # Determine how to read the data based on the type of data_or_path
                if isinstance(data_or_path, str):
                    # It's a file path
                    if not os.path.exists(data_or_path):
                        logger.warning(f"Cannot upload model: File not found at {data_or_path}")
                        return {'success': False, 'error': 'File not found'}
                    
                    # Read the file
                    with open(data_or_path, 'rb') as f:
                        file_data = f.read()
                        file_size = len(file_data)
                
                elif hasattr(data_or_path, 'read'):
                    # It's a file-like object (e.g., BytesIO)
                    # Make sure we're at the beginning
                    if hasattr(data_or_path, 'seek'):
                        data_or_path.seek(0)
                    file_data = data_or_path.read()
                    file_size = len(file_data)
                
                else:
                    # Assume it's binary data
                    file_data = data_or_path
                    file_size = len(file_data)
                
                # Upload to Dropbox
                upload_result = self.dbx.files_upload(file_data, dropbox_path, 
                                                    mode=WriteMode.overwrite)
                
                # Update model files map
                self.model_files[model_name] = dropbox_path
                
                # Create shared link for the file
                shared_link = self.dbx.sharing_create_shared_link_with_settings(dropbox_path)
                
                # Convert shared link to direct download link
                download_url = shared_link.url.replace('?dl=0', '?dl=1')
                
                # Get metadata for response
                result = {
                    'success': True,
                    'name': model_name,
                    'path': dropbox_path,
                    'size': file_size,
                    'upload_time': datetime.now().isoformat(),
                    'download_url': download_url
                }
                
                logger.info(f"Uploaded model {model_name} to Dropbox: {dropbox_path}")
                return result
                
            except Exception as e:
                logger.error(f"Error uploading model {model_name}: {e}")
                return {'success': False, 'error': str(e)}
    
    def download_model(self, model_name: str, local_path: str = None) -> Dict[str, Any]:
        """
        Download a model file from Dropbox.
        
        Args:
            model_name: Name of the model file in Dropbox
            local_path: Path where to save the model (if None, uses temp directory)
            
        Returns:
            Dict with model information (success, local_path)
        """
        with self.lock:
            # Sync model files if needed
            current_time = time.time()
            if current_time - self.last_models_sync > self.models_sync_interval:
                self._sync_model_files()
            
            # Find model path
            dropbox_path = self.model_files.get(model_name)
            if not dropbox_path:
                logger.warning(f"Model {model_name} not found in Dropbox")
                return {'success': False, 'error': 'Model not found'}
            
            # Set local path if not provided
            if not local_path:
                local_path = os.path.join(self.temp_dir, model_name)
            
            try:
                # Download file
                self.dbx.files_download_to_file(local_path, dropbox_path)
                
                result = {
                    'success': True,
                    'local_path': local_path,
                    'name': model_name,
                    'size': os.path.getsize(local_path),
                    'download_time': datetime.now().isoformat()
                }
                
                logger.info(f"Downloaded model {model_name} from Dropbox: {local_path}")
                return result
                
            except Exception as e:
                logger.error(f"Error downloading model {model_name}: {e}")
                return {'success': False, 'error': str(e)}
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of all model files in Dropbox.
        
        Returns:
            List of model information dictionaries
        """
        with self.lock:
            # Sync model files
            self._sync_model_files()
            
            if not self.model_files:
                return []
                
            try:
                # Get details for each model
                models = []
                for name, path in self.model_files.items():
                    try:
                        metadata = self.dbx.files_get_metadata(path)
                        shared_link = self.dbx.sharing_create_shared_link_with_settings(path)
                        download_url = shared_link.url.replace('?dl=0', '?dl=1')
                        
                        models.append({
                            'name': name,
                            'path': path,
                            'size': metadata.size,
                            'modified_date': metadata.server_modified.isoformat(),
                            'download_url': download_url
                        })
                    except Exception as e:
                        logger.error(f"Error getting metadata for model {name}: {e}")
                
                return models
                
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                return []
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model file from Dropbox.
        
        Args:
            model_name: Name of the model file to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        with self.lock:
            # Find model path
            dropbox_path = self.model_files.get(model_name)
            if not dropbox_path:
                logger.warning(f"Cannot delete: Model {model_name} not found in Dropbox")
                return False
            
            try:
                # Delete file
                self.dbx.files_delete_v2(dropbox_path)
                
                # Update model files map
                del self.model_files[model_name]
                
                logger.info(f"Deleted model {model_name} from Dropbox")
                return True
                
            except Exception as e:
                logger.error(f"Error deleting model {model_name}: {e}")
                return False

# Module-level singleton instance
_dropbox_storage = None

def init_dropbox_storage(api_key: str, db_filename: str = "interactions.db", models_folder_name: str = "backdoor_models"):
    """
    Initialize Dropbox storage.
    
    Args:
        api_key: Dropbox API key
        db_filename: Name of the database file in Dropbox
        models_folder_name: Name of the folder to store models in
        
    Returns:
        DropboxStorage: The initialized storage instance
    """
    global _dropbox_storage
    
    if _dropbox_storage is None:
        _dropbox_storage = DropboxStorage(api_key, db_filename, models_folder_name)
    
    return _dropbox_storage

def get_dropbox_storage():
    """
    Get the Dropbox storage instance.
    
    Returns:
        DropboxStorage: The singleton storage instance
        
    Raises:
        RuntimeError: If dropbox storage hasn't been initialized
    """
    if _dropbox_storage is None:
        raise RuntimeError("Dropbox storage not initialized. Call init_dropbox_storage() first.")
    
    return _dropbox_storage
