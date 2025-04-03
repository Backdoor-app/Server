"""
Learning package for the Backdoor AI application.

This package contains modules for:
- Text preprocessing (preprocessor.py)
- Intent classification (intent_classifier.py)
- Model training and evaluation (trainer.py)
"""

from .preprocessor import preprocess_text, extract_features, ensure_nltk_resources
from .intent_classifier import IntentClassifier
from .trainer import (
    should_retrain, 
    trigger_retraining, 
    train_new_model, 
    get_current_model_version,
    clean_old_models
)

__all__ = [
    # Preprocessor
    'preprocess_text', 
    'extract_features',
    'ensure_nltk_resources',
    
    # Classifier
    'IntentClassifier',
    
    # Trainer
    'should_retrain',
    'trigger_retraining',
    'train_new_model',
    'get_current_model_version',
    'clean_old_models'
]