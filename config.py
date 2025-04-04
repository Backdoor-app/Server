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
DROPBOX_API_KEY = os.getenv("DROPBOX_API_KEY", "sl.u.AFruoRTQC3QAqNsx7Vyz4uGG1cWTG3t-6Fks2Ual1wHmIXiPCqb3wdBpHySEHUBrD8JdncsoY3jHyoQJnRAFyTDYA89F0VuFKKq0TQDaXqXzO_AQrWD9csYMqJEan0KKtMkvj_8nBA-qkcBNqo7ZVNorSLtUE1u9POW5ksf-r9o5IlMWrxbbNMnJr8WZVDaZE2eophrHFXUHQLtr5dV6Wch4x5i3AlG5BTxBfNZLERToPbio3kMQS22MQut9QOYRXNEKJp2yy6Cuei7VLJKr-0iGjiduJVnzWpZI3VmMZOykzetI7qUOHmEtOnJDtXZRiYV2nbNKAe9vwLSCccuJWRERaoq7AGL_rScRf-qqzUIzbuqqb8ANJVZoqHUOx6w0KJL4mkGfx_dff5xqbkQ7KaChWp7iezoqopQwzG6LQVh31SDvhcDrLvZ5YwWXRnge5aF9i7bpI2MAG6y1cOalNd_jR1UycChhuNo7df0EEaPUuIgs7A-X7p1cSiipM8BJOGWuZuHfLQMb29UH8ZGdFoVNiSm4GfJgnHxmuxqf_9jkafK0WPiDaJsf-jKY8O_AhJxgDpUchXi8Q-kMn-5OoHIw57tZ5eKlPh_uWg90SpkjmUcVuU18Ww9RA0AExI7N3QfeEHMdXGXzNUOJtShxVtdGX21YglyNtPqmRFjAiqZMFCsxaLY6kD6w21KgDvlBI2zUpscgT5LACiEC_kM1av8i04bLd7HvnRuijKhcS6uKj_yr7kTG5b-7jKoAVojNw7K4_7xl12yuvZJkS1F4H2IDt3wdlPz3BQYBQkmVM5TewVe11AKfQGqdpT_Mhw1sinvkyyOHBy3lBwvxV9TDfHuPFvSDryEwmO74c_Jkr2CEMpeX7KrKeZ5VKzFL35OfWXKla4czUzr1gtrmO-nCSFbkwqHTYxgdWjbK7jnhasLxm7zuZaAwcPrgl5mlJ6f3sWBFbDeUqHYG0kRtcoq3PthTWBzTJHRV8PnDKJW394zKRxLDsxhi75tZWGRnvm7Wa29QARhFmhegfSttTEwZEUz2J0tL3Lewn6zdqFFEGz-7n_VvlkAPuwKfhqXCf92exlu4AtgBxNr4wLUp6CPvKi27FsaYNcozgwaHE__CPjgLmLNhJee0iFzwISlG5xwh4qicOSM91HmYYrrgvvO5ifrABNg23IEI8BbExGOcVHWXFyfscvE7U72K2x6LEs2lQppJbBZ-BcRFxEvvt1Gp7ZHwuVupP9z45tMIlGBYNTcm80wfp4BDR5fsUh5IrANMRIz2WGJAxTECltjHb0cTNjOsZb7cRkfg8imUzVcMVJRimaeikWzNuRAWKeRU791nyu1OCoYiZIO3DJI0QB5E9Wmlu_2YZ7YjMvOSZzAUnJ16DtK3fuHqJ0fFFSUyyG639Nx84ld8fDD6hUril_CZ1vIPfWhRlzhSHFEWsFMSb8OuAw")
DROPBOX_DB_FILENAME = os.getenv("DROPBOX_DB_FILENAME", "backdoor_ai_db.db")
DROPBOX_MODELS_FOLDER = os.getenv("DROPBOX_MODELS_FOLDER", "backdoor_models")

# Dropbox Sync Settings
DROPBOX_DB_SYNC_INTERVAL = int(os.getenv("DROPBOX_DB_SYNC_INTERVAL", "60"))  # Seconds
DROPBOX_MODELS_SYNC_INTERVAL = int(os.getenv("DROPBOX_MODELS_SYNC_INTERVAL", "300"))  # Seconds

# Storage Mode (dropbox or local)
STORAGE_MODE = "dropbox" if DROPBOX_ENABLED else "local"

# Base model name (used to identify the model in Dropbox/local storage)
BASE_MODEL_NAME = "model_1.0.0.mlmodel"
