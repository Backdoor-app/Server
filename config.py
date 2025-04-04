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

# Google Drive Integration Settings
GOOGLE_DRIVE_ENABLED = os.getenv("GOOGLE_DRIVE_ENABLED", "True").lower() in ["true", "1", "yes"]
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", os.path.join(BASE_DIR, "google_credentials.json"))
GOOGLE_DB_FILENAME = os.getenv("GOOGLE_DB_FILENAME", "backdoor_ai_db.db")
GOOGLE_MODELS_FOLDER = os.getenv("GOOGLE_MODELS_FOLDER", "backdoor_models")

# Google Drive Sync Settings
GOOGLE_DB_SYNC_INTERVAL = int(os.getenv("GOOGLE_DB_SYNC_INTERVAL", "60"))  # Seconds
GOOGLE_MODELS_SYNC_INTERVAL = int(os.getenv("GOOGLE_MODELS_SYNC_INTERVAL", "300"))  # Seconds

# Storage Mode (google_drive or local)
STORAGE_MODE = "google_drive" if GOOGLE_DRIVE_ENABLED else "local"
# Base Model Google Drive File ID (Public link)
# This is the file ID from the shared Google Drive link
BASE_MODEL_DRIVE_FILE_ID = "1xrV5BoUqFppd6Wc-MANUJwOFzCvK4BXV"  # Provided by user - public link to CoreML model  # Replace with the actual file ID
