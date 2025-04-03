"""
Google Drive integration for Backdoor AI server.

This module provides functionality to:
- Store and retrieve SQLite database files on Google Drive
- Upload and download ML model files
- Handle authentication and synchronization
"""

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import tempfile
import time
import threading
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class DriveStorage:
    """Handles Google Drive storage operations for the Backdoor AI server."""
    
    def __init__(self, credentials_path: str, db_filename: str = "interactions.db", models_folder_name: str = "backdoor_models"):
        """
        Initialize Google Drive storage.
        
        Args:
            credentials_path: Path to the Google service account credentials JSON file
            db_filename: Name of the database file in Google Drive
            models_folder_name: Name of the folder to store models in
        """
        self.credentials_path = credentials_path
        self.db_filename = db_filename
        self.models_folder_name = models_folder_name
        
        # File tracking
        self.db_file_id = None
        self.models_folder_id = None
        self.model_files = {}  # Mapping of model names to file IDs
        
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
        
        # Connect to Google Drive
        self.drive = self._authenticate()
        
        # Initialize resources
        self._initialize()
        
    def _authenticate(self) -> GoogleDrive:
        """
        Authenticate with Google Drive using service account credentials.
        
        Returns:
            GoogleDrive: Authenticated GoogleDrive instance
        
        Raises:
            Exception: If authentication fails
        """
        try:
            gauth = GoogleAuth()
            
            # Check if credentials file exists
            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError(f"Google credentials file not found: {self.credentials_path}")
                
            # Use service account authentication
            scope = ['https://www.googleapis.com/auth/drive']
            gauth.ServiceAuth(service_account_file=self.credentials_path)
            
            return GoogleDrive(gauth)
            
        except Exception as e:
            logger.error(f"Google Drive authentication failed: {e}")
            raise
            
    def _initialize(self) -> None:
        """Initialize Google Drive resources (find or create DB file and models folder)."""
        try:
            # Check if database file exists
            self._find_db_file()
            
            # Check if models folder exists
            self._find_or_create_models_folder()
            
            # Download database if it exists
            if self.db_file_id:
                self._download_db()
                
            # Sync model file list
            self._sync_model_files()
            
            logger.info("Google Drive storage initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive storage: {e}")
            raise
    
    def _find_db_file(self) -> bool:
        """
        Find the database file in Google Drive.
        
        Returns:
            bool: True if file was found, False otherwise
        """
        try:
            file_list = self.drive.ListFile({'q': f"title='{self.db_filename}' and trashed=false"}).GetList()
            if file_list:
                self.db_file_id = file_list[0]['id']
                logger.info(f"Found database file in Google Drive: {self.db_file_id}")
                return True
            else:
                logger.info(f"Database file '{self.db_filename}' not found in Google Drive, will create on first upload")
                return False
        except Exception as e:
            logger.error(f"Error finding database file: {e}")
            return False
    
    def _find_or_create_models_folder(self) -> bool:
        """
        Find or create the models folder in Google Drive.
        
        Returns:
            bool: True if folder was found or created, False otherwise
        """
        try:
            # Check if folder already exists
            folder_list = self.drive.ListFile({
                'q': f"title='{self.models_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            
            if folder_list:
                self.models_folder_id = folder_list[0]['id']
                logger.info(f"Found models folder in Google Drive: {self.models_folder_id}")
                return True
            
            # Create folder if it doesn't exist
            folder_metadata = {
                'title': self.models_folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.drive.CreateFile(folder_metadata)
            folder.Upload()
            
            self.models_folder_id = folder['id']
            logger.info(f"Created models folder in Google Drive: {self.models_folder_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error finding or creating models folder: {e}")
            return False
    
    def _sync_model_files(self) -> bool:
        """
        Sync the list of model files from Google Drive.
        
        Returns:
            bool: True if sync was successful, False otherwise
        """
        if not self.models_folder_id:
            return False
            
        try:
            file_list = self.drive.ListFile({
                'q': f"'{self.models_folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            
            # Update model files map
            self.model_files = {file['title']: file['id'] for file in file_list}
            self.last_models_sync = time.time()
            
            logger.info(f"Synced {len(self.model_files)} model files from Google Drive")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing model files: {e}")
            return False
    
    def _download_db(self) -> bool:
        """
        Download the database from Google Drive.
        
        Returns:
            bool: True if download was successful, False otherwise
        """
        if not self.db_file_id:
            return False
            
        try:
            drive_file = self.drive.CreateFile({'id': self.db_file_id})
            drive_file.GetContentFile(self.local_db_path)
            self.last_db_sync = time.time()
            
            logger.info(f"Downloaded database from Google Drive: {self.local_db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading database: {e}")
            return False
    
    def get_db_path(self) -> str:
        """
        Get the local path to the database file, downloading it from Drive if needed.
        
        Returns:
            str: Path to the local database file
        """
        with self.lock:
            # Check if we need to sync from Google Drive
            current_time = time.time()
            if current_time - self.last_db_sync > self.db_sync_interval:
                if self.db_file_id:
                    self._download_db()
                else:
                    self._find_db_file()
                    if self.db_file_id:
                        self._download_db()
            
            return self.local_db_path
    
    def upload_db(self) -> bool:
        """
        Upload the database to Google Drive.
        
        Returns:
            bool: True if upload was successful, False otherwise
        """
        with self.lock:
            if not os.path.exists(self.local_db_path):
                logger.warning(f"Cannot upload database: File not found at {self.local_db_path}")
                return False
                
            try:
                if self.db_file_id:
                    # Update existing file
                    drive_file = self.drive.CreateFile({'id': self.db_file_id})
                else:
                    # Create new file
                    drive_file = self.drive.CreateFile({'title': self.db_filename})
                
                # Set content and upload
                drive_file.SetContentFile(self.local_db_path)
                drive_file.Upload()
                
                # Update file ID and sync time
                self.db_file_id = drive_file['id']
                self.last_db_sync = time.time()
                
                logger.info(f"Uploaded database to Google Drive: {self.db_file_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error uploading database: {e}")
                return False
    
    def upload_model(self, local_path: str, model_name: str) -> Dict[str, Any]:
        """
        Upload a model file to Google Drive.
        
        Args:
            local_path: Path to the local model file
            model_name: Name to use for the model in Google Drive
            
        Returns:
            Dict with model information (id, name, download_url)
        """
        with self.lock:
            if not os.path.exists(local_path):
                logger.warning(f"Cannot upload model: File not found at {local_path}")
                return {'success': False, 'error': 'File not found'}
                
            if not self.models_folder_id:
                self._find_or_create_models_folder()
                if not self.models_folder_id:
                    return {'success': False, 'error': 'Models folder not available'}
            
            try:
                # Check if model with same name already exists
                model_id = self.model_files.get(model_name)
                
                if model_id:
                    # Update existing file
                    drive_file = self.drive.CreateFile({'id': model_id})
                else:
                    # Create new file
                    drive_file = self.drive.CreateFile({
                        'title': model_name,
                        'parents': [{'id': self.models_folder_id}]
                    })
                
                # Set content and upload
                drive_file.SetContentFile(local_path)
                drive_file.Upload()
                
                # Update model files map
                model_id = drive_file['id']
                self.model_files[model_name] = model_id
                
                # Get metadata for response
                result = {
                    'success': True,
                    'id': model_id,
                    'name': model_name,
                    'size': os.path.getsize(local_path),
                    'upload_time': datetime.now().isoformat(),
                    'download_url': drive_file['alternateLink']
                }
                
                logger.info(f"Uploaded model {model_name} to Google Drive: {model_id}")
                return result
                
            except Exception as e:
                logger.error(f"Error uploading model {model_name}: {e}")
                return {'success': False, 'error': str(e)}
    
    def download_model(self, model_name: str, local_path: str = None) -> Dict[str, Any]:
        """
        Download a model file from Google Drive.
        
        Args:
            model_name: Name of the model file in Google Drive
            local_path: Path where to save the model (if None, uses temp directory)
            
        Returns:
            Dict with model information (success, local_path)
        """
        with self.lock:
            # Sync model files if needed
            current_time = time.time()
            if current_time - self.last_models_sync > self.models_sync_interval:
                self._sync_model_files()
            
            # Find model ID
            model_id = self.model_files.get(model_name)
            if not model_id:
                logger.warning(f"Model {model_name} not found in Google Drive")
                return {'success': False, 'error': 'Model not found'}
            
            # Set local path if not provided
            if not local_path:
                local_path = os.path.join(self.temp_dir, model_name)
            
            try:
                # Download file
                drive_file = self.drive.CreateFile({'id': model_id})
                drive_file.GetContentFile(local_path)
                
                result = {
                    'success': True,
                    'local_path': local_path,
                    'name': model_name,
                    'size': os.path.getsize(local_path),
                    'download_time': datetime.now().isoformat()
                }
                
                logger.info(f"Downloaded model {model_name} from Google Drive: {local_path}")
                return result
                
            except Exception as e:
                logger.error(f"Error downloading model {model_name}: {e}")
                return {'success': False, 'error': str(e)}
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of all model files in Google Drive.
        
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
                for name, file_id in self.model_files.items():
                    file_info = self.drive.CreateFile({'id': file_id})
                    file_info.FetchMetadata()
                    
                    models.append({
                        'name': name,
                        'id': file_id,
                        'size': int(file_info['fileSize']) if 'fileSize' in file_info else 0,
                        'created_date': file_info['createdDate'],
                        'modified_date': file_info['modifiedDate'],
                        'download_url': file_info['alternateLink']
                    })
                
                return models
                
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                return []
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model file from Google Drive.
        
        Args:
            model_name: Name of the model file to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        with self.lock:
            # Find model ID
            model_id = self.model_files.get(model_name)
            if not model_id:
                logger.warning(f"Cannot delete: Model {model_name} not found in Google Drive")
                return False
            
            try:
                # Delete file
                drive_file = self.drive.CreateFile({'id': model_id})
                drive_file.Delete()
                
                # Update model files map
                del self.model_files[model_name]
                
                logger.info(f"Deleted model {model_name} from Google Drive")
                return True
                
            except Exception as e:
                logger.error(f"Error deleting model {model_name}: {e}")
                return False

# Module-level singleton instance
_drive_storage = None

def init_drive_storage(credentials_path: str, db_filename: str = "interactions.db", models_folder_name: str = "backdoor_models"):
    """
    Initialize Google Drive storage.
    
    Args:
        credentials_path: Path to the credentials JSON file
        db_filename: Name of the database file in Google Drive
        models_folder_name: Name of the folder to store models in
        
    Returns:
        DriveStorage: The initialized storage instance
    """
    global _drive_storage
    
    if _drive_storage is None:
        _drive_storage = DriveStorage(credentials_path, db_filename, models_folder_name)
    
    return _drive_storage

def get_drive_storage():
    """
    Get the Google Drive storage instance.
    
    Returns:
        DriveStorage: The singleton storage instance
        
    Raises:
        RuntimeError: If drive storage hasn't been initialized
    """
    if _drive_storage is None:
        raise RuntimeError("Google Drive storage not initialized. Call init_drive_storage() first.")
    
    return _drive_storage
