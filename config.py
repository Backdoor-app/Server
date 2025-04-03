"""
Configuration settings for the Backdoor AI Learning Server
"""
import os

# API Keys
API_KEY = os.getenv("API_KEY", "rnd_2DfFj1QmKeAWcXF5u9Z0oV35kBiN")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "rnd_2DfFj1QmKeAWcXF5u9Z0oV35kBiN")

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