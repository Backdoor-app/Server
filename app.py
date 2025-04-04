"""
Backdoor AI Learning Server - Main Application

This module contains the main Flask application and API endpoints for:
- Collecting user interaction data
- Uploading user-trained models
- Providing access to trained models
- Collecting application statistics
"""

from flask import Flask, request, jsonify, send_file, render_template_string, redirect
from flask_cors import CORS
import os
import json
import sqlite3
import schedule
import time
import threading
import signal
from datetime import datetime
import logging
import subprocess
import nltk
import tempfile  # For temporary directory management

# Import configuration
import config
import sys  # Required for sys.exit in scheduler

# Import from packages
from utils.db_helpers import init_db, store_interactions, store_uploaded_model
from learning import (
    ensure_nltk_resources, 
    IntentClassifier,
    should_retrain, 
    trigger_retraining, 
    train_new_model,
    get_current_model_version,
    clean_old_models
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backdoor_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create a thread lock for database operations
db_lock = threading.RLock()

# Create temporary directory for NLTK data
temp_nltk_dir = tempfile.mkdtemp()
nltk.data.path.append(temp_nltk_dir)

logger.info("Operating in memory-only mode with Dropbox storage - no local directories needed")

# This section has been removed as we're not using local directories anymore
# NLTK uses temporary directory configured above

# Ensure NLTK resources are available
ensure_nltk_resources()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Check if Dropbox integration is enabled
if config.DROPBOX_ENABLED:
    try:
        # Import Dropbox storage module
        from utils.dropbox_storage import init_dropbox_storage, get_dropbox_storage
        
        # Initialize Dropbox storage with API key
        dropbox_storage = init_dropbox_storage(
            config.DROPBOX_API_KEY,
            config.DROPBOX_DB_FILENAME,
            config.DROPBOX_MODELS_FOLDER
        )
        
        logger.info(f"Dropbox storage initialized successfully")
        logger.info(f"Using database: {config.DROPBOX_DB_FILENAME}")
        logger.info(f"Using models folder: {config.DROPBOX_MODELS_FOLDER}")
        
    except Exception as e:
        logger.error(f"Failed to initialize Dropbox storage: {e}")
        logger.warning("Falling back to local storage")
        # Disable Dropbox mode
        config.DROPBOX_ENABLED = False
        # Also update the compatibility variable
        config.GOOGLE_DRIVE_ENABLED = False

# Initialize database on startup
init_db(config.DB_PATH)

import os
# Import Dropbox storage functionality
if config.DROPBOX_ENABLED:
    from utils.dropbox_storage import init_dropbox_storage, get_dropbox_storage
    from learning.trainer_dropbox import check_base_model_in_dropbox

# Check for base model in Dropbox
try:
    base_model_found = False
    if config.DROPBOX_ENABLED:
        # Check if base model exists in Dropbox
        base_model_available = check_base_model_in_dropbox()
        if base_model_available:
            logger.info(f"Base model '{config.BASE_MODEL_NAME}' found in Dropbox and is available for use")
            
            # Ensure the model reference is in the database
            from utils.db_helpers import get_connection
            with get_connection(config.DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM model_versions WHERE version = '1.0.0'")
                if cursor.fetchone()[0] == 0:
                    # Add base model reference to database
                    base_model_path = f"dropbox:/{config.DROPBOX_MODELS_FOLDER}/{config.BASE_MODEL_NAME}"
                    cursor.execute("""
                        INSERT INTO model_versions 
                        (version, path, accuracy, training_data_size, training_date)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        '1.0.0',  # version
                        base_model_path,  # path
                        0.92,  # accuracy
                        1000,  # training_data_size
                        datetime.now().isoformat()  # training_date
                    ))
                    conn.commit()
                    logger.info(f"Added base model reference to database: {base_model_path}")
            base_model_found = True
        else:
            logger.warning(f"Base model '{config.BASE_MODEL_NAME}' not found in Dropbox. Please upload it to your Dropbox folder.")
    else:
        # Check if base model exists locally
        base_model_path = os.path.join(config.MODEL_DIR, config.BASE_MODEL_NAME)
        if os.path.exists(base_model_path):
            logger.info(f"Base model found at {base_model_path} and is available for use")
            
            # Ensure the model reference is in the database
            from utils.db_helpers import get_connection
            with get_connection(config.DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM model_versions WHERE version = '1.0.0'")
                if cursor.fetchone()[0] == 0:
                    # Add base model reference to database
                    cursor.execute("""
                        INSERT INTO model_versions 
                        (version, path, accuracy, training_data_size, training_date)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        '1.0.0',  # version
                        base_model_path,  # path
                        0.92,  # accuracy
                        1000,  # training_data_size
                        datetime.now().isoformat()  # training_date
                    ))
                    conn.commit()
                    logger.info(f"Added base model reference to database: {base_model_path}")
            base_model_found = True
        else:
            logger.warning(f"Base model not found at {base_model_path}. Please place your model file in the models directory.")
    
    if not base_model_found:
        logger.warning("No base model found. Model training will not work correctly until a base model is provided.")
except Exception as e:
    logger.error(f"Error checking base model: {e}")
# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/api/ai/learn', methods=['POST'])
def collect_data():
    """
    API endpoint for collecting user interaction data from devices.
    """
    try:
        data = request.json
        device_id = data.get('deviceId', 'unknown')
        logger.info(f"Received learning data from device: {device_id}")
        
        if not data or 'interactions' not in data:
            return jsonify({'success': False, 'message': 'Invalid data format'}), 400
        
        # Store interactions in database with lock
        with db_lock:
            store_interactions(config.DB_PATH, data)
            
        # Get latest model info to return to client
        latest_model = get_latest_model_info()
        
        return jsonify({
            'success': True,
            'message': 'Data received successfully',
            'latestModelVersion': latest_model['version'],
            'modelDownloadURL': f"https://{request.host}/api/ai/models/{latest_model['version']}"
        })
        
    except Exception as e:
        logger.error(f"Error processing learning data: {e}")
        return jsonify({'success': False, 'message': f'Error: {e}'}), 500

@app.route('/api/ai/upload-model', methods=['POST'])
def upload_model():
    """
    API endpoint for uploading user-trained CoreML models.
    
    These models will be incorporated into an ensemble model on the server.
    Uploads directly to Dropbox without creating local files.
    """
    try:
        # Check if file is included in the request
        if 'model' not in request.files:
            return jsonify({'success': False, 'message': 'No model file provided'}), 400
        
        model_file = request.files['model']
        
        # Check if a valid file was selected
        if model_file.filename == '':
            return jsonify({'success': False, 'message': 'No model file selected'}), 400
        
        # Get device ID and other metadata
        device_id = request.form.get('deviceId', 'unknown')
        app_version = request.form.get('appVersion', 'unknown')
        description = request.form.get('description', '')
        
        # Ensure the file is a CoreML model
        if not model_file.filename.endswith('.mlmodel'):
            return jsonify({'success': False, 'message': 'File must be a CoreML model (.mlmodel)'}), 400
        
        # Generate a unique filename
        timestamp = int(datetime.now().timestamp())
        unique_filename = f"model_upload_{device_id}_{timestamp}.mlmodel"
        
        # Get Dropbox storage
        from utils.dropbox_storage import get_dropbox_storage
        dropbox_storage = get_dropbox_storage()
        
        # Read the file content directly to memory and upload to Dropbox
        model_data = model_file.read()
        file_size = len(model_data)
        
        # Upload directly to Dropbox
        upload_result = dropbox_storage.upload_model(model_data, unique_filename)
        
        if not upload_result.get('success', False):
            return jsonify({'success': False, 'message': f"Error uploading to Dropbox: {upload_result.get('error', 'Unknown error')}"}), 500
        
        # Get the Dropbox path for reference
        dropbox_path = upload_result.get('path', '')
        
        # Store model metadata in database with lock
        with db_lock:
            model_id = store_uploaded_model(
                config.DB_PATH, 
                device_id=device_id,
                app_version=app_version,
                description=description,
                file_path=f"dropbox:{dropbox_path}",  # Store the Dropbox path as reference
                file_size=file_size,
                original_filename=model_file.filename
            )
        
        # Trigger async model retraining if conditions are met
        if should_retrain(config.DB_PATH):
            # Use Thread with daemon=True to ensure it terminates when main thread exits
            retraining_thread = threading.Thread(
                target=trigger_retraining, 
                args=(config.DB_PATH,), 
                daemon=True
            )
            retraining_thread.start()
            retraining_status = "Model retraining triggered"
        else:
            retraining_status = "Model will be incorporated in next scheduled training"
        
        # Return success response
        latest_model = get_latest_model_info()
        return jsonify({
            'success': True,
            'message': f'Model uploaded successfully. {retraining_status}',
            'modelId': model_id,
            'latestModelVersion': latest_model['version'],
            'modelDownloadURL': f"https://{request.host}/api/ai/models/{latest_model['version']}"
        })
        
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        return jsonify({'success': False, 'message': f'Error: {e}'}), 500

@app.route('/api/ai/models/<version>', methods=['GET'])
def get_model(version):
    """
    API endpoint for downloading a specific model version.
    
    Streams directly from Dropbox without creating local files.
    """
    try:
        # If this is the base model version, serve from memory cache
        if version == '1.0.0':
            from utils.model_download import get_base_model_buffer
            model_buffer = get_base_model_buffer()
            
            if model_buffer:
                logger.info(f"Serving base model version {version} from memory")
                model_buffer.seek(0)  # Ensure we're at the beginning of the buffer
                return send_file(
                    model_buffer,
                    mimetype='application/octet-stream',
                    as_attachment=True,
                    download_name=f"model_{version}.mlmodel"
                )
        
        # For other versions, get streaming URL from Dropbox
        if config.DROPBOX_ENABLED:
            try:
                from utils.dropbox_storage import get_dropbox_storage
                dropbox_storage = get_dropbox_storage()
                model_name = f"model_{version}.mlmodel"
                
                # Get model information (including direct download URL)
                model_info = dropbox_storage.get_model_stream(model_name)
                
                if model_info and model_info.get('success'):
                    download_url = model_info.get('download_url')
                    
                    if download_url:
                        # Redirect to the direct download URL
                        logger.info(f"Redirecting to Dropbox direct download for model {version}")
                        return redirect(download_url)
                    
                    # If we couldn't get a direct URL, try downloading to memory and serving
                    memory_download = dropbox_storage.download_model_to_memory(model_name)
                    if memory_download and memory_download.get('success'):
                        logger.info(f"Serving model version {version} from memory buffer")
                        model_buffer = memory_download.get('model_buffer')
                        model_buffer.seek(0)
                        return send_file(
                            model_buffer,
                            mimetype='application/octet-stream',
                            as_attachment=True,
                            download_name=f"model_{version}.mlmodel"
                        )
            except Exception as e:
                logger.error(f"Error retrieving model from Dropbox: {e}")
        
        # Check if model_path is a stream or memory reference from get_model_path
        from utils.db_helpers import get_model_path
        model_path = get_model_path(config.DB_PATH, version)
        
        if model_path:
            # Handle stream URL references
            if model_path.startswith('stream:'):
                stream_url = model_path.split(':', 1)[1]
                logger.info(f"Redirecting to stream URL for model {version}")
                return redirect(stream_url)
                
            # Handle memory buffer references
            elif model_path.startswith('memory:'):
                model_name = model_path.split(':', 1)[1]
                from utils.dropbox_storage import get_dropbox_storage
                dropbox_storage = get_dropbox_storage()
                memory_info = dropbox_storage.download_model_to_memory(model_name)
                
                if memory_info and memory_info.get('success'):
                    logger.info(f"Serving model version {version} from memory")
                    model_buffer = memory_info.get('model_buffer')
                    model_buffer.seek(0)
                    return send_file(
                        model_buffer,
                        mimetype='application/octet-stream',
                        as_attachment=True,
                        download_name=f"model_{version}.mlmodel"
                    )
            
            # For legacy support, check if it's a local path
            elif os.path.exists(model_path):
                logger.info(f"Serving model version {version} from local path {model_path}")
                return send_file(model_path, mimetype='application/octet-stream')
                
        logger.warning(f"Model version {version} not found")
        return jsonify({'success': False, 'message': 'Model not found'}), 404
        
    except Exception as e:
        logger.error(f"Error in get_model: {e}")
        return jsonify({'success': False, 'message': f'Error: {e}'}), 500

@app.route('/api/ai/latest-model', methods=['GET'])
def latest_model():
    """
    API endpoint for getting information about the latest model.
    """
    model_info = get_latest_model_info()
    
    return jsonify({
        'success': True,
        'message': 'Latest model info',
        'latestModelVersion': model_info['version'],
        'modelDownloadURL': f"https://{request.host}/api/ai/models/{model_info['version']}"
    })

@app.route('/api/ai/stats', methods=['GET'])
def get_stats():
    """
    API endpoint for getting system statistics.
    """
    conn = None
    try:
        with db_lock:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Get total interactions
            cursor.execute("SELECT COUNT(*) FROM interactions")
            total_interactions = cursor.fetchone()[0]
            
            # Get unique devices
            cursor.execute("SELECT COUNT(DISTINCT device_id) FROM interactions")
            unique_devices = cursor.fetchone()[0]
            
            # Get average feedback rating
            cursor.execute("SELECT AVG(rating) FROM feedback")
            avg_rating = cursor.fetchone()[0] or 0
            
            # Get top intents
            cursor.execute("""
                SELECT detected_intent, COUNT(*) as count 
                FROM interactions 
                GROUP BY detected_intent 
                ORDER BY count DESC 
                LIMIT 5
            """)
            top_intents = [{"intent": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            # Get models statistics
            cursor.execute("SELECT COUNT(*) FROM model_versions")
            total_models = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM uploaded_models WHERE incorporation_status = 'incorporated'")
            incorporated_models = cursor.fetchone()[0]
        
        # Get latest model info
        model_info = get_latest_model_info()
        
        return jsonify({
            'success': True,
            'stats': {
                'totalInteractions': total_interactions,
                'uniqueDevices': unique_devices,
                'averageFeedbackRating': round(avg_rating, 2),
                'topIntents': top_intents,
                'latestModelVersion': model_info['version'],
                'lastTrainingDate': model_info.get('training_date', 'Unknown'),
                'totalModels': total_models,
                'incorporatedUserModels': incorporated_models
            }
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'success': False, 'message': f'Error: {e}'}), 500
    finally:
        if conn:
            conn.close()

def get_latest_model_info():
    """
    Get information about the latest model version.
    
    Uses database to get info with fallback to default values.
    No local file storage used.
    
    Returns:
        dict: Model information
    """
    try:
        # Try to get model info from database
        from utils.db_helpers import get_connection
        with get_connection(config.DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version, path, accuracy, training_data_size, training_date 
                FROM model_versions 
                ORDER BY created_at DESC LIMIT 1
            """)
            result = cursor.fetchone()
            
            if result:
                return {
                    'version': result[0],
                    'path': result[1],
                    'accuracy': result[2],
                    'training_data_size': result[3],
                    'training_date': result[4],
                    'is_ensemble': False  # Default, could be updated from ensemble_models table
                }
    except Exception as e:
        logger.error(f"Error getting latest model from database: {e}")
    
    # If database query fails or no models found, return default info
    default_info = {
        'version': '1.0.0',
        'path': 'dropbox:/model_1.0.0.mlmodel',  # Reference to Dropbox path
        'training_date': datetime.now().isoformat(),
        'accuracy': 0.0,
        'training_data_size': 0,
        'is_ensemble': False
    }
    
    return default_info

def train_model_job():
    """
    Scheduled job to train a new model using the latest data.
    """
    try:
        logger.info("Starting scheduled model training")
        
        # Check if we should run training
        if should_retrain(config.DB_PATH):
            with db_lock:
                new_version = train_new_model(config.DB_PATH)
            
            # Clean up old models to save space in Dropbox
            clean_old_models_dropbox(config.MAX_MODELS_TO_KEEP)
            
            logger.info(f"Model training completed. New version: {new_version}")
        else:
            logger.info("Scheduled training skipped - not enough new data or models")
    except Exception as e:
        logger.error(f"Model training failed: {e}")

def run_scheduler():
    """
    Run the scheduler for periodic tasks.
    
    Note: Signal handling moved to main thread to prevent errors.
    """
    
    # Schedule the training job
    schedule.every().day.at("02:00").do(train_model_job)
    
    # Add periodic model cleanup from Dropbox
    schedule.every().week.do(lambda: clean_old_models_dropbox(config.MAX_MODELS_TO_KEEP))
    
    logger.info("Scheduler started")
    
    # Main scheduler loop
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except Exception as e:
            logger.error(f"Error in scheduler: {e}")
            time.sleep(300)  # Wait longer after an error

# Start the scheduler in a daemon thread
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# Register signal handlers in the main thread only when running as a script
# This avoids issues with signal handlers in threads
if __name__ == '__main__':
    def shutdown_handler(signum, frame):
        logger.info("Received shutdown signal, exiting application")
        sys.exit(0)
        
    # Register signal handlers in the main thread
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    logger.info("Signal handlers registered in main thread")

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the server is running properly.
    """
    try:
        # Check if database is accessible
        conn = None
        try:
            conn = sqlite3.connect(config.DB_PATH)
            conn.execute("SELECT 1")
            db_status = "healthy"
        except Exception as e:
            db_status = f"unhealthy: {e}"
        finally:
            if conn:
                conn.close()
                
        # Check if model directory is accessible    
        model_status = "healthy" if os.access(config.MODEL_DIR, os.R_OK | os.W_OK) else "unhealthy: permission denied"
        
        # Check scheduler status
        scheduler_status = "running" if scheduler_thread and scheduler_thread.is_alive() else "not running"
        
        # Check model files
        try:
            model_files = [f for f in os.listdir(config.MODEL_DIR) if f.endswith(".mlmodel")]
            model_count = len(model_files)
        except Exception:
            model_count = "error"
            
        return jsonify({
            'status': 'up',
            'database': db_status,
            'models': model_status,
            'scheduler': scheduler_status,
            'model_count': model_count,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# API Documentation page
@app.route('/', methods=['GET'])
def api_documentation():
    """Serve the API documentation page."""
    # The HTML template remains unchanged - it's a large static template
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Backdoor AI API Documentation</title>
        <style>
            :root {
                --primary-color: #2563eb;
                --primary-hover: #1e40af;
                --secondary-color: #64748b;
                --bg-color: #f8fafc;
                --card-bg: #ffffff;
                --code-bg: #f1f5f9;
                --border-color: #e2e8f0;
                --text-color: #334155;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: var(--bg-color);
                margin: 0;
                padding: 20px;
            }
            
            .container {
                max-width: 1000px;
                margin: 0 auto;
            }
            
            header {
                margin-bottom: 40px;
                text-align: center;
                padding-bottom: 20px;
                border-bottom: 1px solid var(--border-color);
            }
            
            h1 {
                color: var(--primary-color);
                margin-bottom: 10px;
            }
            
            h2 {
                margin-top: 40px;
                padding-bottom: 10px;
                border-bottom: 1px solid var(--border-color);
            }
            
            h3 {
                margin-top: 25px;
                color: var(--secondary-color);
            }
            
            .endpoint-card {
                background-color: var(--card-bg);
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                padding: 20px;
                margin-bottom: 30px;
                position: relative;
            }
            
            .method {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
                color: white;
                margin-right: 10px;
            }
            
            .get {
                background-color: #22c55e;
            }
            
            .post {
                background-color: #3b82f6;
            }
            
            .path {
                font-family: monospace;
                font-size: 18px;
                font-weight: 600;
                vertical-align: middle;
            }
            
            .copy-btn {
                position: absolute;
                top: 20px;
                right: 20px;
                background-color: var(--primary-color);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                cursor: pointer;
                font-size: 14px;
                transition: background-color 0.2s;
            }
            
            .copy-btn:hover {
                background-color: var(--primary-hover);
            }
            
            pre {
                background-color: var(--code-bg);
                padding: 15px;
                border-radius: 6px;
                overflow: auto;
                font-family: monospace;
                font-size: 14px;
            }
            
            code {
                font-family: monospace;
                background-color: var(--code-bg);
                padding: 2px 5px;
                border-radius: 4px;
                font-size: 14px;
            }
            
            .description {
                margin: 15px 0;
            }
            
            /* Removed auth-info styles as authentication is no longer required */
            
            .request-example, .response-example {
                margin-top: 15px;
            }
            
            .parameters {
                margin-top: 15px;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            
            th, td {
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid var(--border-color);
            }
            
            th {
                background-color: var(--code-bg);
                font-weight: 600;
            }
            
            footer {
                margin-top: 60px;
                text-align: center;
                padding-top: 20px;
                border-top: 1px solid var(--border-color);
                color: var(--secondary-color);
                font-size: 14px;
            }
            
            .tooltip {
                position: relative;
                display: inline-block;
            }
            
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 140px;
                background-color: #555;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 150%;
                left: 50%;
                margin-left: -75px;
                opacity: 0;
                transition: opacity 0.3s;
            }
            
            .tooltip .tooltiptext::after {
                content: "";
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -5px;
                border-width: 5px;
                border-style: solid;
                border-color: #555 transparent transparent transparent;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Backdoor AI API Documentation</h1>
                <p>API documentation for the Backdoor AI Learning Server</p>
            </header>
            
            <h2>Endpoints</h2>
            
            <!-- POST /api/ai/learn -->
            <div class="endpoint-card">
                <span class="method post">POST</span>
                <span class="path">/api/ai/learn</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/api/ai/learn')">Copy URL</button>
                
                <div class="description">
                    <p>Submit interaction data from devices to be used for model training. Returns information about the latest model version.</p>
                </div>
                
                <!-- No authentication required -->
                
                <div class="request-example">
                    <h3>Request Example</h3>
                    <pre>{
  "deviceId": "device_123",
  "appVersion": "1.2.0",
  "modelVersion": "1.0.0",
  "osVersion": "iOS 15.0",
  "interactions": [
    {
      "id": "int_abc123",
      "timestamp": "2023-06-15T14:30:00Z",
      "userMessage": "Turn on the lights",
      "aiResponse": "Turning on the lights",
      "detectedIntent": "light_on",
      "confidenceScore": 0.92,
      "feedback": {
        "rating": 5,
        "comment": "Perfect response"
      }
    }
  ]
}</pre>
                </div>
                
                <div class="response-example">
                    <h3>Response Example</h3>
                    <pre>{
  "success": true,
  "message": "Data received successfully",
  "latestModelVersion": "1.0.1712052481",
  "modelDownloadURL": "https://yourdomain.com/api/ai/models/1.0.1712052481"
}</pre>
                </div>
            </div>
            
            <!-- POST /api/ai/upload-model -->
            <div class="endpoint-card">
                <span class="method post">POST</span>
                <span class="path">/api/ai/upload-model</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/api/ai/upload-model')">Copy URL</button>
                
                <div class="description">
                    <p>Upload a CoreML model trained on your device to be combined with other models on the server. The server will create an ensemble model incorporating multiple uploaded models.</p>
                </div>
                
                <!-- No authentication required -->
                
                <div class="request-example">
                    <h3>Request Format</h3>
                    <p>This endpoint requires a <code>multipart/form-data</code> request with the following fields:</p>
                    <table>
                        <tr>
                            <th>Field</th>
                            <th>Type</th>
                            <th>Description</th>
                        </tr>
                        <tr>
                            <td>model</td>
                            <td>File</td>
                            <td>The CoreML (.mlmodel) file to upload</td>
                        </tr>
                        <tr>
                            <td>deviceId</td>
                            <td>String</td>
                            <td>The unique identifier of the uploading device</td>
                        </tr>
                        <tr>
                            <td>appVersion</td>
                            <td>String</td>
                            <td>The version of the app sending the model</td>
                        </tr>
                        <tr>
                            <td>description</td>
                            <td>String</td>
                            <td>Optional description of the model</td>
                        </tr>
                    </table>
                </div>
                
                <div class="response-example">
                    <h3>Response Example</h3>
                    <pre>{
  "success": true,
  "message": "Model uploaded successfully. Model will be incorporated in next scheduled training",
  "modelId": "d290f1ee-6c54-4b01-90e6-d701748f0851",
  "latestModelVersion": "1.0.1712052481",
  "modelDownloadURL": "https://yourdomain.com/api/ai/models/1.0.1712052481"
}</pre>
                </div>
                
                <div class="description">
                    <h3>Model Processing</h3>
                    <p>After models are uploaded:</p>
                    <ul>
                        <li>They are stored on the server and queued for processing</li>
                        <li>When enough models are uploaded (3+) or after a time threshold, retraining is triggered</li>
                        <li>The server combines all uploaded models with its base model using ensemble techniques</li>
                        <li>The resulting model is available through the standard model endpoints</li>
                    </ul>
                </div>
            </div>
            
            <!-- GET /api/ai/models/{version} -->
            <div class="endpoint-card">
                <span class="method get">GET</span>
                <span class="path">/api/ai/models/{version}</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/api/ai/models/1.0.0')">Copy URL</button>
                
                <div class="description">
                    <p>Download a specific model version. Returns the CoreML model file.</p>
                </div>
                
                <!-- No authentication required -->
                
                <div class="parameters">
                    <h3>URL Parameters</h3>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Description</th>
                        </tr>
                        <tr>
                            <td>version</td>
                            <td>The version of the model to download (e.g., "1.0.0")</td>
                        </tr>
                    </table>
                </div>
                
                <div class="response-example">
                    <h3>Response</h3>
                    <p>Binary file (CoreML model) or error message if model not found.</p>
                </div>
            </div>
            
            <!-- GET /api/ai/latest-model -->
            <div class="endpoint-card">
                <span class="method get">GET</span>
                <span class="path">/api/ai/latest-model</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/api/ai/latest-model')">Copy URL</button>
                
                <div class="description">
                    <p>Get information about the latest trained model. Returns the version and download URL.</p>
                </div>
                
                <!-- No authentication required -->
                
                <div class="response-example">
                    <h3>Response Example</h3>
                    <pre>{
  "success": true,
  "message": "Latest model info",
  "latestModelVersion": "1.0.1712052481",
  "modelDownloadURL": "https://yourdomain.com/api/ai/models/1.0.1712052481"
}</pre>
                </div>
            </div>
            
            <!-- GET /api/ai/stats -->
            <div class="endpoint-card">
                <span class="method get">GET</span>
                <span class="path">/api/ai/stats</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/api/ai/stats')">Copy URL</button>
                
                <div class="description">
                    <p>Get statistics about the collected data and model training. For admin use only.</p>
                </div>
                
                <!-- No authentication required -->
                
                <div class="response-example">
                    <h3>Response Example</h3>
                    <pre>{
  "success": true,
  "stats": {
    "totalInteractions": 1250,
    "uniqueDevices": 48,
    "averageFeedbackRating": 4.32,
    "topIntents": [
      {"intent": "light_on", "count": 325},
      {"intent": "temperature_query", "count": 214},
      {"intent": "music_play", "count": 186},
      {"intent": "weather_query", "count": 142},
      {"intent": "timer_set", "count": 95}
    ],
    "latestModelVersion": "1.0.1712052481",
    "lastTrainingDate": "2025-04-01T02:00:00Z",
    "totalModels": 5,
    "incorporatedUserModels": 12
  }
}</pre>
                </div>
            </div>
            
            <!-- GET /health -->
            <div class="endpoint-card">
                <span class="method get">GET</span>
                <span class="path">/health</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/health')">Copy URL</button>
                
                <div class="description">
                    <p>Health check endpoint to verify the server is running properly. Checks database and model storage accessibility.</p>
                </div>
                
                <div class="response-example">
                    <h3>Response Example</h3>
                    <pre>{
  "status": "up",
  "database": "healthy",
  "models": "healthy",
  "scheduler": "running",
  "model_count": 5,
  "timestamp": "2025-04-01T10:15:30Z"
}</pre>
                </div>
            </div>
            
            <footer>
                <p>Backdoor AI Learning Server &copy; 2025</p>
            </footer>
        </div>
        
        <script>
            function copyToClipboard(text) {
                navigator.clipboard.writeText(text).then(function() {
                    var buttons = document.getElementsByClassName('copy-btn');
                    for (var i = 0; i < buttons.length; i++) {
                        buttons[i].textContent = 'Copy URL';
                    }
                    
                    var clickedButton = event.target;
                    var originalText = clickedButton.textContent;
                    clickedButton.textContent = 'Copied!';
                    
                    setTimeout(function() {
                        clickedButton.textContent = originalText;
                    }, 2000);
                }, function(err) {
                    console.error('Could not copy text: ', err);
                });
            }
        </script>
    </body>
    </html>
    '''
    
    return render_template_string(html_template)

# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == '__main__':
    # Display version info
    pip_version = subprocess.check_output(["pip", "--version"]).decode("utf-8").strip()
    logger.info(f"Using pip version: {pip_version}")
    
    # Log startup information
    logger.info(f"Starting Backdoor AI Learning Server on port {config.PORT}")
    logger.info(f"Data directory: {config.BASE_DIR}")
    logger.info(f"Model directory: {config.MODEL_DIR}")
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=config.PORT, debug=False)