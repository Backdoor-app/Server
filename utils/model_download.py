"""
Utility for downloading the base model from Dropbox.

This module provides functionality to:
- Download and verify the base model file
- Detect when the model is missing and download it automatically
- Use the model as the base for intent classification
"""

import os
import logging
import requests
import tempfile
import time
import shutil
from pathlib import Path
import config

logger = logging.getLogger(__name__)

def download_file_from_dropbox(url, destination):
    """
    Download a file from Dropbox using a direct download URL
    
    Args:
        url: Dropbox direct download URL
        destination: Local path where the file should be saved
        
    Returns:
        bool: True if download successful, False otherwise
    """
    # Ensure the URL is a direct download link
    if '?dl=0' in url:
        url = url.replace('?dl=0', '?dl=1')
    elif '?dl=' not in url:
        url = url + '?dl=1'
    
    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    
    try:
        logger.info(f"Downloading base model from Dropbox")
        # Start streaming download
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            start_time = time.time()
            last_log_time = start_time
            
            with open(temp_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192*16):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress periodically (every 5 seconds)
                        current_time = time.time()
                        if current_time - last_log_time > 5:
                            elapsed = current_time - start_time
                            speed = downloaded / elapsed if elapsed > 0 else 0
                            percent = (downloaded / total_size * 100) if total_size > 0 else 0
                            logger.info(f"Download progress: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB) - {speed/(1024*1024):.2f} MB/s")
                            last_log_time = current_time
        
        # Move the temp file to the destination
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(temp_file, destination)
        
        logger.info(f"Base model downloaded successfully to {destination}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading base model: {e}")
        # Clean up temp file
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return False

def ensure_base_model():
    """
    Ensure the base model exists, downloading it if necessary
    
    Returns:
        bool: True if the model is available (either existed or was downloaded), False otherwise
    """
    # Base model path
    model_version = "1.0.0"  # Base version
    model_path = os.path.join(config.MODEL_DIR, f"model_{model_version}.mlmodel")
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:  # >1MB = real model
        logger.info(f"Base model already exists at {model_path}")
        return True
    
    # Check if we have a Dropbox URL to download from
    if not hasattr(config, 'BASE_MODEL_DROPBOX_URL') or not config.BASE_MODEL_DROPBOX_URL:
        logger.warning("No Dropbox URL defined for base model. Set BASE_MODEL_DROPBOX_URL in config.")
        return False
    
    # Download the model
    download_success = download_file_from_dropbox(
        config.BASE_MODEL_DROPBOX_URL, 
        model_path
    )
    
    if download_success:
        # Also place the model in the location expected by GitHub workflow
        github_model_path = os.path.join(config.BASE_DIR, "model", "coreml_model.mlmodel")
        os.makedirs(os.path.dirname(github_model_path), exist_ok=True)
        
        # Use copy instead of move to keep both locations updated
        try:
            shutil.copy(model_path, github_model_path)
            logger.info(f"Copied base model to GitHub workflow location: {github_model_path}")
        except Exception as e:
            logger.error(f"Failed to copy model to GitHub workflow location: {e}")
    
    return download_success
