"""
Configuration settings for the Backdoor AI Learning Server
"""
import os

# Server settings
PORT = int(os.getenv("PORT", 10000))

# Storage paths
BASE_DIR = os.getenv("RENDER_DISK_PATH", "/opt/render/project")
DB_PATH = os.path.join(BASE_DIR, "data", "interactions.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")
NLTK_DATA_PATH = os.path.join(BASE_DIR, "nltk_data")
UPLOADED_MODELS_DIR = os.path.join(MODEL_DIR, "uploaded")

# NLTK Resources
NLTK_RESOURCES = ['punkt', 'stopwords', 'wordnet']

# Model training settings
MIN_TRAINING_DATA = 50
MAX_MODELS_TO_KEEP = 5  # Keep only the most recent N models
RETRAINING_THRESHOLDS = {
    'pending_models': 3,           # Retrain if there are at least 3 pending models
    'hours_since_last_training': 12, # Retrain if it's been 12+ hours and we have pending models
    'new_interactions': 100        # Retrain if we have 100+ new interactions and pending models
}

# Text processing settings
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# Ensemble model settings
BASE_MODEL_WEIGHT = 2.0    # Weight of the base model in the ensemble
USER_MODEL_WEIGHT = 1.0    # Weight of each user-contributed model

# Database lock timeout (seconds)
DB_LOCK_TIMEOUT = 60

# Model naming
MODEL_VERSION_PREFIX = "1.0."  # Base prefix for model versions

# Dropbox Integration Settings
DROPBOX_ENABLED = os.getenv("DROPBOX_ENABLED", "True").lower() in ["true", "1", "yes"]
DROPBOX_API_KEY = os.getenv("DROPBOX_API_KEY", "sl.u.AFr644NtvwSXMgahi8lLvhJKeiMS4Vmk3nq0AlYjiagi0iLUZHbkWfUM2ITVdu5840l2olzscEivBNt5ps43j0")
DROPBOX_DB_FILENAME = os.getenv("DROPBOX_DB_FILENAME", "backdoor_ai_db.db")
DROPBOX_MODELS_FOLDER = os.getenv("DROPBOX_MODELS_FOLDER", "backdoor_models")

# Dropbox Sync Settings
DROPBOX_DB_SYNC_INTERVAL = int(os.getenv("DROPBOX_DB_SYNC_INTERVAL", "60"))  # Seconds
DROPBOX_MODELS_SYNC_INTERVAL = int(os.getenv("DROPBOX_MODELS_SYNC_INTERVAL", "300"))  # Seconds

# Storage Mode (dropbox or local)
STORAGE_MODE = "dropbox" if DROPBOX_ENABLED else "local"

# Base Model Dropbox URL
# This is the direct download URL to the CoreML model
BASE_MODEL_DROPBOX_URL = "https://www.dropbox.com/scl/fi/2xarhyii46tr9amkqh764/coreml_model.mlmodel?rlkey=j3cxmpjhxj8bbwzw11j1hy54c&st=ak8zbpxp&dl=1"
