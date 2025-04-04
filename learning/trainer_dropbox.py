"""
Functions for Dropbox-based model management in memory-only mode.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import config

logger = logging.getLogger(__name__)

def clean_old_models_dropbox(keep_newest: int = config.MAX_MODELS_TO_KEEP) -> None:
    """
    Delete old model files from Dropbox to prevent storage issues
    
    Args:
        keep_newest: Number of most recent models to keep
    """
    try:
        # Get dropbox storage
        from utils.dropbox_storage import get_dropbox_storage
        dropbox_storage = get_dropbox_storage()
        
        # List all models
        model_list = dropbox_storage.list_models()
        
        # Filter to just the versioned models (not uploaded user models)
        model_files = []
        for model in model_list:
            filename = model.get('name', '')
            if filename.startswith("model_") and filename.endswith(".mlmodel"):
                version = filename.replace("model_", "").replace(".mlmodel", "")
                try:
                    # Extract timestamp from version (assuming format like 1.0.1712052481)
                    timestamp = int(version.split(".")[-1])
                    model_files.append({
                        'filename': filename,
                        'version': version,
                        'timestamp': timestamp,
                        'path': model.get('path', ''),
                        'name': model.get('name', '')
                    })
                except (ValueError, IndexError):
                    # Skip files with unexpected version format
                    continue
                    
        # Skip cleanup if we don't have more than the keep limit
        if len(model_files) <= keep_newest:
            logger.info(f"No model cleanup needed - only {len(model_files)} models found")
            return
            
        # Sort by timestamp (newest first)
        model_files.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Keep the newest N models, delete the rest
        models_to_delete = model_files[keep_newest:]
        for model in models_to_delete:
            # Delete model from Dropbox
            try:
                result = dropbox_storage.delete_model(model['name'])
                if result:
                    logger.info(f"Deleted old model from Dropbox: {model['version']}")
                else:
                    logger.warning(f"Failed to delete model from Dropbox: {model['version']}")
            except Exception as e:
                logger.error(f"Error deleting model {model['version']} from Dropbox: {e}")
                
        logger.info(f"Cleaned up {len(models_to_delete)} old models, kept {keep_newest} newest")
    
    except Exception as e:
        logger.error(f"Error cleaning up old models from Dropbox: {e}")
