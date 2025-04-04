"""
Test script for the model download functionality.

This script tests the model download from Google Drive and
ensures that the base model is properly set up.
"""

import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.append('.')

# Import the necessary modules
import config
from utils.model_download import ensure_base_model

def test_model_download():
    """Test downloading the model from Google Drive"""
    try:
        # Make sure model directories exist
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(os.path.join(config.BASE_DIR, "model"), exist_ok=True)
        
        # Check if model files already exist
        app_model_path = os.path.join(config.MODEL_DIR, f"model_1.0.0.mlmodel") 
        github_model_path = os.path.join(config.BASE_DIR, "model", "coreml_model.mlmodel")
        
        logger.info(f"Checking for existing model at: {app_model_path}")
        if os.path.exists(app_model_path):
            size_mb = os.path.getsize(app_model_path) / (1024 * 1024)
            logger.info(f"App model exists: {size_mb:.2f} MB")
        else:
            logger.info("App model does not exist")
            
        logger.info(f"Checking for existing model at: {github_model_path}")
        if os.path.exists(github_model_path):
            size_mb = os.path.getsize(github_model_path) / (1024 * 1024)
            logger.info(f"GitHub workflow model exists: {size_mb:.2f} MB")
        else:
            logger.info("GitHub workflow model does not exist")
        
        # Check if we have a file ID configured
        if not hasattr(config, 'BASE_MODEL_DRIVE_FILE_ID') or not config.BASE_MODEL_DRIVE_FILE_ID:
            logger.error("No Google Drive file ID defined for base model. Update config.py.")
            return False
        
        # Try to download/ensure the model
        logger.info("Attempting to ensure the base model is available...")
        result = ensure_base_model()
        
        if result:
            logger.info("Success! Base model is available.")
            
            # Check model files again
            if os.path.exists(app_model_path):
                size_mb = os.path.getsize(app_model_path) / (1024 * 1024)
                logger.info(f"App model now exists: {size_mb:.2f} MB")
            
            if os.path.exists(github_model_path):
                size_mb = os.path.getsize(github_model_path) / (1024 * 1024)
                logger.info(f"GitHub workflow model now exists: {size_mb:.2f} MB")
                
            return True
        else:
            logger.error("Failed to ensure base model is available")
            return False
    
    except Exception as e:
        logger.error(f"Error in test: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting model download test")
    test_model_download()
    logger.info("Test completed")
