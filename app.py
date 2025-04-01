from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import sqlite3
import schedule
import time
import threading
from datetime import datetime
import logging
import subprocess
import nltk
import stat  # Added for permission debugging
from utils.db_helpers import init_db, store_interactions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("backdoor_ai.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set the base directory for persistent storage
BASE_DIR = os.getenv("RENDER_DISK_PATH", "/opt/render/project")
DB_PATH = os.path.join(BASE_DIR, "data", "interactions.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Debug: Check permissions of the base directory
def check_permissions(path):
    try:
        stats = os.stat(path)
        permissions = stat.filemode(stats.st_mode)
        logger.info(f"Permissions for {path}: {permissions}")
    except Exception as e:
        logger.error(f"Cannot check permissions for {path}: {e}")

check_permissions(BASE_DIR)

# Set NLTK data path to persistent disk
NLTK_DATA_PATH = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Download NLTK data if not present
try:
    for resource in ['punkt', 'stopwords', 'wordnet']:
        resource_path = os.path.join(NLTK_DATA_PATH, resource)
        if not os.path.exists(resource_path):
            logger.info(f"Downloading NLTK resource: {resource} to {NLTK_DATA_PATH}")
            nltk.download(resource, download_dir=NLTK_DATA_PATH, quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data to {NLTK_DATA_PATH}: {e}. Using fallback.")
    raise  # Re-raise for debugging purposes

# Initialize Flask app
app = Flask(__name__)
CORS(app)

PORT = int(os.getenv("PORT", 10000))
init_db(DB_PATH)
API_KEY = os.getenv("API_KEY", "rnd_2DfFj1QmKeAWcXF5u9Z0oV35kBiN")

@app.route('/api/ai/learn', methods=['POST'])
def collect_data():
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    try:
        data = request.json
        logger.info(f"Received learning data from device: {data.get('deviceId', 'unknown')}")
        if not data or 'interactions' not in data:
            return jsonify({'success': False, 'message': 'Invalid data format'}), 400
        store_interactions(DB_PATH, data)
        latest_model = get_latest_model_info()
        return jsonify({
            'success': True,
            'message': 'Data received successfully',
            'latestModelVersion': latest_model['version'],
            'modelDownloadURL': f"https://{request.host}/api/ai/models/{latest_model['version']}"
        })
    except Exception as e:
        logger.error(f"Error processing learning data: {str(e)}")
        return jsonify({'success': False, 'messageV2': f'Error: {str(e)}'}), 500

@app.route('/api/ai/models/<version>', methods=['GET'])
def get_model(version):
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    model_path = os.path.join(MODEL_DIR, f"model_{version}.mlmodel")
    if os.path.exists(model_path):
        logger.info(f"Serving model version {version}")
        return send_file(model_path, mimetype='application/octet-stream')
    else:
        logger.warning(f"Model version {version} not found")
        return jsonify({'success': False, 'message': 'Model not found'}), 404

@app.route('/api/ai/latest-model', methods=['GET'])
def latest_model():
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    model_info = get_latest_model_info()
    return jsonify({
        'success': True,
        'message': 'Latest model info',
        'latestModelVersion': model_info['version'],
        'modelDownloadURL': f"https://{request.host}/api/ai/models/{model_info['version']}"
    })

@app.route('/api/ai/stats', methods=['GET'])
def get_stats():
    admin_key = os.getenv("ADMIN_API_KEY", "rnd_2DfFj1QmKeAWcXF5u9Z0oV35kBiN")
    if request.headers.get('X-Admin-Key') != admin_key:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM interactions")
        total_interactions = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT device_id) FROM interactions")
        unique_devices = cursor.fetchone()[0]
        cursor.execute("SELECT AVG(rating) FROM feedback")
        avg_rating = cursor.fetchone()[0] or 0
        cursor.execute("""
            SELECT detected_intent, COUNT(*) as count 
            FROM interactions 
            GROUP BY detected_intent 
            ORDER BY count DESC 
            LIMIT 5
        """)
        top_intents = [{"intent": row[0], "count": row[1]} for row in cursor.fetchall()]
        conn.close()
        model_info = get_latest_model_info()
        return jsonify({
            'success': True,
            'stats': {
                'totalInteractions': total_interactions,
                'uniqueDevices': unique_devices,
                'averageFeedbackRating': round(avg_rating, 2),
                'topIntents': top_intents,
                'latestModelVersion': model_info['version'],
                'lastTrainingDate': model_info.get('training_date', 'Unknown')
            }
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

def get_latest_model_info():
    info_path = os.path.join(MODEL_DIR, "latest_model.json")
    if not os.path.exists(info_path):
        default_info = {
            'version': '1.0.0',
            'path': os.path.join(MODEL_DIR, 'model_1.0.0.mlmodel'),
            'training_date': datetime.now().isoformat()
        }
        with open(info_path, 'w') as f:
            json.dump(default_info, f)
        return default_info
    with open(info_path, 'r') as f:
        return json.load(f)

def train_model_job():
    # Lazy import to avoid circular import
    from learning.trainer import train_new_model
    try:
        logger.info("Starting scheduled model training")
        new_version = train_new_model(DB_PATH)
        logger.info(f"Model training completed. New version: {new_version}")
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")

def run_scheduler():
    schedule.every().day.at("02:00").do(train_model_job)
    while True:
        schedule.run_pending()
        time.sleep(60)

scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.daemon = True
scheduler_thread.start()

if __name__ == '__main__':
    pip_version = subprocess.check_output(["pip", "--version"]).decode("utf-8").strip()
    logger.info(f"Using pip version: {pip_version}")
    logger.info("Starting Backdoor AI Learning Server on Render")
    app.run(host='0.0.0.0', port=PORT, debug=False)