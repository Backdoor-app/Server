"""
Intent classification module for the Backdoor AI learning system.

This module provides the IntentClassifier class which encapsulates:
- Training intent classification models
- Creating ensemble models from user-uploaded models
- Predictions using trained models
- Model serialization/deserialization
- CoreML model conversion
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union, Callable

# ML libraries
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import coremltools as ct

# Local imports
from .preprocessor import preprocess_text, extract_features
import config

logger = logging.getLogger(__name__)

class IntentClassifier:
    """
    Intent classification model for detecting user intents from messages.
    
    This class handles all aspects of model training, prediction, and serialization.
    """
    
    def __init__(self):
        """Initialize a new, untrained IntentClassifier."""
        self.vectorizer = None
        self.model = None
        self.classes = None
        self.is_ensemble = False
        self.component_models = {}
        self.model_version = None
        self.training_date = None
        self.accuracy = None
        self.training_data_size = 0
        
    @property
    def is_trained(self) -> bool:
        """
        Check if the model has been trained.
        
        Returns:
            bool: True if the model is trained, False otherwise
        """
        return self.model is not None and self.vectorizer is not None
    
    def train(self, 
              data: pd.DataFrame, 
              user_message_col: str = 'user_message',
              intent_col: str = 'detected_intent',
              weight_col: Optional[str] = 'weight',
              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the intent classifier on interaction data.
        
        Args:
            data: DataFrame containing training data
            user_message_col: Column name for user messages
            intent_col: Column name for intent labels
            weight_col: Column name for sample weights, if any
            test_size: Proportion of data to use for testing
            
        Returns:
            Dict containing training results (accuracy, report, etc.)
        """
        if len(data) < config.MIN_TRAINING_DATA:
            raise ValueError(f"Insufficient training data: {len(data)} samples (minimum: {config.MIN_TRAINING_DATA})")
        
        # Preprocess text data
        logger.info(f"Preprocessing {len(data)} messages for training")
        data['processed_message'] = data[user_message_col].apply(preprocess_text)
        
        # Prepare data
        X = data['processed_message']
        y = data[intent_col]
        
        # Use sample weights if available
        weights = None
        if weight_col and weight_col in data.columns:
            weights = data[weight_col].values
        
        # Split data
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights,
            test_size=test_size,
            random_state=42,
            stratify=y if len(set(y)) > 1 else None
        )
        
        logger.info(f"Training data: {len(X_train)} samples, Test data: {len(X_test)} samples")
        
        # Extract features
        self.vectorizer, X_train_vec = extract_features(
            X_train, 
            max_features=config.MAX_FEATURES, 
            ngram_range=config.NGRAM_RANGE
        )
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train_vec, y_train, sample_weight=w_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_vec)
        self.accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store classes and metadata
        self.classes = self.model.classes_.tolist()
        self.training_data_size = len(X_train)
        self.training_date = datetime.now().isoformat()
        timestamp = int(datetime.now().timestamp())
        self.model_version = f"{config.MODEL_VERSION_PREFIX}{timestamp}"
        
        logger.info(f"Model trained with accuracy: {self.accuracy:.4f}")
        return {
            'accuracy': self.accuracy,
            'report': report,
            'classes': self.classes,
            'model_version': self.model_version,
            'training_data_size': self.training_data_size
        }
    
    def create_ensemble(self, 
                       uploaded_models: List[Dict[str, Any]],
                       base_weight: float = config.BASE_MODEL_WEIGHT) -> bool:
        """
        Create an ensemble model incorporating uploaded user models.
        
        Args:
            uploaded_models: List of dictionaries with model information
            base_weight: Weight of the base model in the ensemble
            
        Returns:
            bool: True if ensemble creation succeeded, False otherwise
        """
        if not self.is_trained:
            logger.error("Cannot create ensemble: Base model not trained")
            return False
            
        if not uploaded_models:
            logger.warning("No uploaded models provided for ensemble")
            return False
            
        try:
            logger.info(f"Creating ensemble with {len(uploaded_models)} uploaded models")
            
            # Initialize ensemble components with our base model
            estimators = [('base', self.model)]
            self.component_models = {'base': 'Base model'}
            
            # Add user models
            for idx, uploaded_model in enumerate(uploaded_models):
                model_id = uploaded_model.get('id', f'user{idx}')
                try:
                    # Convert uploaded model to scikit-learn model (simplified representation)
                    user_model = RandomForestClassifier(n_estimators=50, random_state=42)
                    
                    # In a real implementation, we would properly convert or integrate the model
                    # Here we're using a placeholder that we'll train on our data
                    # and assign appropriate weights in the ensemble
                    
                    # Add to ensemble
                    estimator_name = f'user{idx}'
                    estimators.append((estimator_name, user_model))
                    
                    # Store metadata
                    self.component_models[estimator_name] = {
                        'id': model_id,
                        'device_id': uploaded_model.get('device_id', 'unknown'),
                        'original_file': uploaded_model.get('original_filename', 'unknown')
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to add model {model_id} to ensemble: {e}")
                    continue
            
            # Create the ensemble using VotingClassifier with appropriate weights
            # Set weights to give more importance to the base model
            weights = [base_weight] + [config.USER_MODEL_WEIGHT] * (len(estimators) - 1)
            
            self.model = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights
            )
            
            self.is_ensemble = True
            logger.info(f"Successfully created ensemble with {len(estimators)} models")
            return True
            
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            return False
            
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict intent from text input.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (predicted intent, confidence score)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Transform to feature vector
        feature_vector = self.vectorizer.transform([processed_text])
        
        # Get prediction and probability
        intent = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        # Find the index of the predicted class
        class_idx = self.classes.index(intent)
        confidence = float(probabilities[class_idx])
        
        return intent, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict intents for multiple text inputs.
        
        Args:
            texts: List of input texts to classify
            
        Returns:
            List of tuples with (predicted intent, confidence score)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Transform to feature vectors
        feature_vectors = self.vectorizer.transform(processed_texts)
        
        # Get predictions and probabilities
        intents = self.model.predict(feature_vectors)
        probabilities = self.model.predict_proba(feature_vectors)
        
        # Combine predictions with confidence scores
        results = []
        for i, intent in enumerate(intents):
            class_idx = self.classes.index(intent)
            confidence = float(probabilities[i][class_idx])
            results.append((intent, confidence))
            
        return results
        
    def save(self, model_dir: str) -> Dict[str, Any]:
        """
        Save the trained model to disk.
        
        Args:
            model_dir: Directory to save the model files
            
        Returns:
            Dict with model information
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Define paths for saved files
        sklearn_path = os.path.join(model_dir, f"intent_classifier_{self.model_version}.joblib")
        info_path = os.path.join(model_dir, f"model_info_{self.model_version}.json")
        coreml_path = os.path.join(model_dir, f"model_{self.model_version}.mlmodel")
        latest_info_path = os.path.join(model_dir, "latest_model.json")
        
        # Save scikit-learn model
        joblib.dump((self.vectorizer, self.model), sklearn_path)
        logger.info(f"Saved sklearn model to {sklearn_path}")
        
        # Create model info
        model_info = {
            'version': self.model_version,
            'accuracy': self.accuracy,
            'training_data_size': self.training_data_size,
            'training_date': self.training_date,
            'is_ensemble': self.is_ensemble,
            'component_models': len(self.component_models) if self.component_models else 1,
            'classes': self.classes,
            'sklearn_path': sklearn_path,
            'coreml_path': coreml_path
        }
        
        # Save model info
        with open(info_path, 'w') as f:
            json.dump(model_info, f)
        
        # Also save as latest model info
        with open(latest_info_path, 'w') as f:
            json.dump(model_info, f)
            
        logger.info(f"Saved model info to {info_path}")
        
        # Convert to CoreML
        try:
            self._convert_to_coreml(coreml_path)
            logger.info(f"Saved CoreML model to {coreml_path}")
        except Exception as e:
            logger.error(f"CoreML conversion failed: {e}")
            model_info['coreml_conversion_error'] = str(e)
            
        return model_info
        
    def _convert_to_coreml(self, output_path: str) -> None:
        """
        Convert the trained model to CoreML format.
        
        Args:
            output_path: Path to save the CoreML model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot convert untrained model")
            
        # Define prediction function that handles preprocessing and prediction
        def predict_intent(text: str) -> Tuple[str, np.ndarray]:
            """Prediction function for CoreML conversion."""
            processed_text = preprocess_text(text)
            vec_text = self.vectorizer.transform([processed_text])
            intent = self.model.predict(vec_text)[0]
            probabilities = self.model.predict_proba(vec_text)[0]
            return intent, probabilities
            
        # Convert to CoreML
        coreml_model = ct.convert(
            predict_intent,
            inputs=[ct.TensorType(shape=(1,), dtype=str)],
            outputs=[
                ct.TensorType(name='intent'),
                ct.TensorType(name='probabilities', dtype=np.float32)
            ],
            classifier_config=ct.ClassifierConfig(self.classes),
            minimum_deployment_target=ct.target.iOS15
        )
        
        # Add metadata
        coreml_model.user_defined_metadata['version'] = self.model_version
        coreml_model.user_defined_metadata['training_date'] = self.training_date
        coreml_model.user_defined_metadata['accuracy'] = str(self.accuracy)
        coreml_model.user_defined_metadata['intents'] = ','.join(self.classes)
        
        if self.is_ensemble:
            coreml_model.user_defined_metadata['is_ensemble'] = 'true'
            coreml_model.user_defined_metadata['ensemble_size'] = str(len(self.component_models))
            coreml_model.user_defined_metadata['component_models'] = json.dumps(
                {k: v for k, v in self.component_models.items() if isinstance(v, str)}
            )
        
        # Save the model
        coreml_model.save(output_path)
        
    @classmethod
    def load(cls, model_path: str) -> 'IntentClassifier':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved sklearn model
            
        Returns:
            IntentClassifier: Loaded classifier instance
        """
        logger.info(f"Loading model from {model_path}")
        try:
            vectorizer, model = joblib.load(model_path)
            
            # Create new instance
            classifier = cls()
            classifier.vectorizer = vectorizer
            classifier.model = model
            
            # Load classes from model
            if hasattr(model, 'classes_'):
                classifier.classes = model.classes_.tolist()
            elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                # For VotingClassifier
                first_estimator = model.estimators_[0]
                if hasattr(first_estimator, 'classes_'):
                    classifier.classes = first_estimator.classes_.tolist()
                    
            # Set ensemble flag
            classifier.is_ensemble = isinstance(model, VotingClassifier)
            
            # Set placeholder for other attributes
            classifier.training_date = datetime.now().isoformat()
            classifier.model_version = os.path.basename(model_path).replace('intent_classifier_', '').replace('.joblib', '')
            
            return classifier
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @classmethod
    def load_from_info(cls, info_path: str) -> 'IntentClassifier':
        """
        Load a trained model using its info file.
        
        Args:
            info_path: Path to the model info JSON file
            
        Returns:
            IntentClassifier: Loaded classifier instance
        """
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
                
            sklearn_path = info.get('sklearn_path')
            if not sklearn_path or not os.path.exists(sklearn_path):
                raise FileNotFoundError(f"Sklearn model not found: {sklearn_path}")
                
            # Load the model
            classifier = cls.load(sklearn_path)
            
            # Restore metadata
            classifier.model_version = info.get('version')
            classifier.training_date = info.get('training_date')
            classifier.accuracy = info.get('accuracy')
            classifier.training_data_size = info.get('training_data_size', 0)
            classifier.is_ensemble = info.get('is_ensemble', False)
            
            if 'classes' in info:
                classifier.classes = info['classes']
                
            return classifier
        except Exception as e:
            logger.error(f"Failed to load model from info: {e}")
            raise