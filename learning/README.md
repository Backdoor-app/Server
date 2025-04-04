# NLP Model Training System

This directory contains the core machine learning components for the Backdoor AI Server.

## Components

- `intent_classifier.py`: The main classification model for intent detection
- `preprocessor.py`: Text preprocessing utilities
- `trainer.py`: Model training and evaluation
- `trainer_dropbox.py`: Dropbox-specific model handling

## Important Notes

### scikit-learn Compatibility

The Core ML tools have a version compatibility requirement with scikit-learn:
- Minimum version: 0.17
- Maximum version: 1.1.2

If you're experiencing errors related to scikit-learn compatibility, ensure you're using version 1.1.2 or earlier:

```bash
pip install scikit-learn==1.1.2
```

### Ensemble Model Creation

The system supports creating ensemble models that combine:
1. The base model (weighted more heavily)
2. User-uploaded models from iOS devices

This allows the system to incorporate user-specific learning while maintaining overall accuracy.

### Model Storage

Models are stored in Dropbox for persistence across deployments. The system will:
1. Check for a base model in Dropbox (named `model_1.0.0.mlmodel`)
2. Train new models using data collected from user interactions
3. Store trained models in Dropbox for later retrieval

## Training Flow

1. User interaction data is collected via the API
2. When sufficient data is available, training is triggered
3. A new model is trained and evaluated
4. User-uploaded models are incorporated if available
5. The model is converted to CoreML format for iOS compatibility
6. The model is stored in Dropbox and made available for download

## Error Handling

The system includes fallback mechanisms in case of model conversion failures:
- It will attempt multiple conversion strategies
- If conversion fails, it will fallback to using the base model
