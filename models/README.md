# Backdoor AI Model Files

This directory contains model files and metadata for the Backdoor AI intent classification system.

## Base CoreML Model

The base CoreML model is automatically downloaded from Google Drive when the server starts. This happens if:

1. The model file doesn't exist locally (`models/model_1.0.0.mlmodel`)
2. The `BASE_MODEL_DRIVE_FILE_ID` is properly configured in `config.py`

### How the Model Download Works

The server includes functionality to:
- Check if the model exists locally
- Download it from Google Drive if needed
- Place it in both locations used by the system (`models/` and `model/`)
- Log progress during download

This ensures a seamless experience where the model is available even in fresh deployments without manual configuration.

### Manual Download Option

If you need to manually download the model:

1. Get it from the Google Drive link associated with the ID in `config.py`
2. Place it at `models/model_1.0.0.mlmodel`
3. Also place a copy at `model/coreml_model.mlmodel` (for GitHub workflow)

## Model Versions and Training

The system assigns version numbers to models using the prefix `1.0.` followed by a timestamp. The base model is simply `1.0.0`.

When user-submitted models are incorporated or retraining occurs, new model versions are created with updated timestamps.

## Directory Structure

- `models/model_1.0.0.mlmodel` - The base CoreML model
- `models/intent_classifier_1.0.0.joblib` - Sklearn companion model
- `models/latest_model.json` - Information about the current model
- `models/model_info_1.0.0.json` - Version-specific metadata
- `models/uploaded/` - Directory for user-uploaded models

## Ensemble Model Creation

When users upload their own models, the system can incorporate them into an ensemble model that uses both the base model and user-uploaded models.

The base model is weighted more heavily in this ensemble to ensure consistent performance.
