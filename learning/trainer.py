import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import joblib
import coremltools as ct
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils.db_helpers import update_model_incorporation_status, get_pending_uploaded_models

logger = logging.getLogger(__name__)

# Download NLTK resources (moved to app.py in previous fix)
# No imports from app.py should be here

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    except:
        tokens = [token for token in tokens if token.isalnum()]
    return ' '.join(tokens)

def should_retrain(db_path):
    """
    Determine if model retraining should be triggered based on:
    1. Number of pending uploaded models
    2. Time since last training
    3. Number of new interactions since last training
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check number of pending uploaded models
        cursor.execute("SELECT COUNT(*) FROM uploaded_models WHERE incorporation_status = 'pending'")
        pending_models_count = cursor.fetchone()[0]
        
        # If we have 3 or more pending models, retrain
        if pending_models_count >= 3:
            logger.info(f"Retraining triggered: {pending_models_count} pending uploaded models")
            conn.close()
            return True
        
        # Check when the last model was trained
        cursor.execute("SELECT MAX(training_date) FROM model_versions")
        last_training = cursor.fetchone()[0]
        
        if last_training:
            last_training_date = datetime.fromisoformat(last_training)
            time_since_training = datetime.now() - last_training_date
            
            # If it's been more than 12 hours since last training and we have pending models
            if time_since_training > timedelta(hours=12) and pending_models_count > 0:
                logger.info(f"Retraining triggered: {pending_models_count} pending models and {time_since_training.total_seconds() / 3600:.1f} hours since last training")
                conn.close()
                return True
            
        # Check if we have enough new interactions since last training
        if last_training:
            cursor.execute("SELECT COUNT(*) FROM interactions WHERE created_at > ?", (last_training,))
            new_interactions = cursor.fetchone()[0]
            
            # If we have 100+ new interactions and at least one pending model
            if new_interactions >= 100 and pending_models_count > 0:
                logger.info(f"Retraining triggered: {pending_models_count} pending models and {new_interactions} new interactions")
                conn.close()
                return True
        
        conn.close()
        return False
    
    except Exception as e:
        logger.error(f"Error checking if retraining is needed: {str(e)}")
        return False

def trigger_retraining(db_path):
    """
    Trigger a model retraining process
    """
    try:
        logger.info("Triggered manual model retraining")
        train_new_model(db_path)
    except Exception as e:
        logger.error(f"Error during triggered retraining: {str(e)}")

def convert_uploaded_coreml_to_sklearn(model_path):
    """
    Convert an uploaded CoreML model back to a scikit-learn compatible model
    This is a simplified approach - in a real system you would need more robust conversion
    """
    try:
        # Load the CoreML model
        coreml_model = ct.models.MLModel(model_path)
        
        # Extract the model specification
        spec = coreml_model.get_spec()
        
        # Create a simple RandomForest model that can be used in an ensemble
        # In a real implementation, you'd extract parameters from the CoreML model
        # Here we're creating a simplified version
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Create a simple vectorizer
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        
        # Since we can't properly extract the original model's parameters,
        # we return a placeholder that will be trained with our data
        return vectorizer, rf_model, spec.description.metadata.userDefined.get('intents', '').split(',')
    
    except Exception as e:
        logger.error(f"Error converting uploaded CoreML model: {str(e)}")
        return None, None, []

def train_new_model(db_path):
    logger.info("Starting model training process")
    conn = sqlite3.connect(db_path)
    query = """
        SELECT i.*, f.rating, f.comment 
        FROM interactions i 
        LEFT JOIN feedback f ON i.id = f.interaction_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    logger.info(f"Loaded {len(df)} interactions for training")
    
    if len(df) < 50:
        logger.warning("Not enough data for training. Need at least 50 interactions.")
        return get_current_model_version()
    
    df['has_feedback'] = df['rating'].notnull()
    df['is_good_feedback'] = (df['rating'] >= 4) & df['has_feedback']
    df['weight'] = 1
    df.loc[df['has_feedback'], 'weight'] = 2
    df.loc[df['is_good_feedback'], 'weight'] = 3
    
    logger.info("Preprocessing text data")
    df['processed_message'] = df['user_message'].apply(preprocess_text)
    
    X = df['processed_message']
    y = df['detected_intent']
    sample_weights = df['weight'].values
    
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights,
        test_size=0.2,
        random_state=42,
        stratify=y if len(set(y)) > 1 else None
    )
    logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Get uploaded models to incorporate into ensemble
    uploaded_models = get_pending_uploaded_models(db_path)
    usable_uploaded_models = []
    user_models_info = []
    
    # Process each uploaded model
    for uploaded_model in uploaded_models:
        model_id = uploaded_model['id']
        try:
            # Update status to processing
            update_model_incorporation_status(db_path, model_id, 'processing')
            
            # Convert the uploaded CoreML model
            logger.info(f"Processing uploaded model {model_id}")
            model_vec, model_clf, intents = convert_uploaded_coreml_to_sklearn(uploaded_model['file_path'])
            
            if model_vec is not None and model_clf is not None:
                # Train the model with our data to ensure compatibility
                # This trains a simplified version on our data that can work in the ensemble
                X_train_user_vec = model_vec.fit_transform(X_train)
                model_clf.fit(X_train_user_vec, y_train)
                
                # Add to our ensemble components
                usable_uploaded_models.append({
                    'id': model_id,
                    'model': model_clf,
                    'vectorizer': model_vec,
                    'device_id': uploaded_model['device_id']
                })
                
                # Add to model info for later storage
                user_models_info.append({
                    'id': model_id,
                    'device_id': uploaded_model['device_id'],
                    'original_file': uploaded_model['original_filename']
                })
                
                logger.info(f"Successfully processed uploaded model {model_id}")
            else:
                logger.warning(f"Could not use uploaded model {model_id}")
                update_model_incorporation_status(db_path, model_id, 'failed')
                
        except Exception as e:
            logger.error(f"Error processing uploaded model {model_id}: {str(e)}")
            update_model_incorporation_status(db_path, model_id, 'failed')
    
    # Train our base model
    logger.info("Training base RandomForest classifier")
    base_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    base_model.fit(X_train_vec, y_train, sample_weight=w_train)
    
    # Create timestamp for model versioning
    timestamp = int(datetime.now().timestamp())
    model_version = f"1.0.{timestamp}"
    
    MODEL_DIR = os.path.join(os.getenv("RENDER_DISK_PATH", "/var/data"), "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create ensemble model if we have uploaded models
    if usable_uploaded_models:
        logger.info(f"Creating ensemble with {len(usable_uploaded_models)} uploaded models")
        
        # Base predict_intent function that will be used if ensemble creation fails
        def base_predict_intent(text):
            processed_text = preprocess_text(text)
            vec_text = vectorizer.transform([processed_text])
            intent = base_model.predict(vec_text)[0]
            probabilities = base_model.predict_proba(vec_text)[0]
            return intent, probabilities
        
        try:
            # Create ensemble using VotingClassifier
            # First, add our base model
            estimators = [('base', base_model)]
            
            # Add user models with proper vectorization pipeline
            for idx, user_model in enumerate(usable_uploaded_models):
                # In a production system, we'd build more sophisticated pipelines
                # or use a better ensemble technique
                estimators.append((f'user{idx}', user_model['model']))
            
            # Create an ensemble model using soft voting (uses prediction probabilities)
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            
            # We need to manually fit the ensemble since we don't have compatible vectorizers
            # This is a simplified approach - in production you'd use pipelines
            ensemble.fit(X_train_vec, y_train)
            
            # In a real implementation, we would have a more sophisticated ensemble method
            # that properly accounts for different vectorizers
            # For now, we'll implement a simple ensemble prediction function
            def ensemble_predict_intent(text):
                processed_text = preprocess_text(text)
                
                # Get base model prediction
                vec_text = vectorizer.transform([processed_text])
                base_intent = base_model.predict(vec_text)[0]
                base_probs = base_model.predict_proba(vec_text)[0]
                
                # Get predictions from user models - simplified approach
                all_intents = []
                all_probs = []
                
                for user_model in usable_uploaded_models:
                    try:
                        user_vec = user_model['vectorizer']
                        user_clf = user_model['model']
                        
                        user_vec_text = user_vec.transform([processed_text])
                        intent = user_clf.predict(user_vec_text)[0]
                        all_intents.append(intent)
                    except:
                        # Skip models that fail
                        pass
                
                # Simple ensemble: use majority vote if available, otherwise base model
                if all_intents:
                    # Count frequencies of predicted intents
                    from collections import Counter
                    intent_counter = Counter(all_intents)
                    most_common = intent_counter.most_common(1)[0]
                    
                    # If the most common intent was predicted by more than one model,
                    # use it, otherwise use the base model's prediction
                    if most_common[1] > 1:
                        ensemble_intent = most_common[0]
                    else:
                        ensemble_intent = base_intent
                else:
                    ensemble_intent = base_intent
                
                # For probabilities, just use the base model's probabilities
                # In a real implementation, we would combine probabilities across models
                return ensemble_intent, base_probs
            
            # Use the ensemble prediction function for the CoreML model
            predict_function = ensemble_predict_intent
            accuracy = ensemble.score(X_test_vec, y_test, sample_weight=w_test)
            logger.info(f"Ensemble model accuracy: {accuracy:.4f}")
            
            # Create description of incorporated models
            incorporated_models = json.dumps(user_models_info)
            
            # Store ensemble info in database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ensemble_models 
                (ensemble_version, description, component_models)
                VALUES (?, ?, ?)
            ''', (
                model_version,
                f"Ensemble model with {len(usable_uploaded_models)} uploaded models",
                incorporated_models
            ))
            conn.commit()
            conn.close()
            
            # Update status for all incorporated models
            for user_model in usable_uploaded_models:
                update_model_incorporation_status(db_path, user_model['id'], 'incorporated', model_version)
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
            logger.info("Falling back to base model only")
            predict_function = base_predict_intent
            accuracy = base_model.score(X_test_vec, y_test, sample_weight=w_test)
    else:
        # Use only the base model
        logger.info("Using base model only (no uploaded models to incorporate)")
        
        def predict_intent(text):
            processed_text = preprocess_text(text)
            vec_text = vectorizer.transform([processed_text])
            intent = base_model.predict(vec_text)[0]
            probabilities = base_model.predict_proba(vec_text)[0]
            return intent, probabilities
        
        predict_function = predict_intent
        accuracy = base_model.score(X_test_vec, y_test, sample_weight=w_test)
    
    # Save the sklearn model
    sklearn_path = os.path.join(MODEL_DIR, f"intent_classifier_{model_version}.joblib")
    joblib.dump((vectorizer, base_model), sklearn_path)
    logger.info(f"Saved sklearn model to {sklearn_path}")
    
    # Convert to CoreML
    logger.info("Converting to CoreML format")
    try:
        class_labels = base_model.classes_.tolist()
        coreml_model = ct.convert(
            predict_function,
            inputs=[ct.TensorType(shape=(1,), dtype=str)],
            outputs=[
                ct.TensorType(name='intent'),
                ct.TensorType(name='probabilities', dtype=np.float32)
            ],
            classifier_config=ct.ClassifierConfig(class_labels),
            minimum_deployment_target=ct.target.iOS15
        )
        
        # Add metadata to track if this is an ensemble
        if usable_uploaded_models:
            coreml_model.user_defined_metadata['is_ensemble'] = 'true'
            coreml_model.user_defined_metadata['ensemble_size'] = str(len(usable_uploaded_models) + 1)
            coreml_model.user_defined_metadata['intents'] = ','.join(class_labels)
        
        coreml_path = os.path.join(MODEL_DIR, f"model_{model_version}.mlmodel")
        coreml_model.save(coreml_path)
        logger.info(f"Saved CoreML model to {coreml_path}")
        
        model_info = {
            'version': model_version,
            'path': coreml_path,
            'accuracy': float(accuracy),
            'training_data_size': len(X_train),
            'training_date': datetime.now().isoformat(),
            'is_ensemble': len(usable_uploaded_models) > 0,
            'component_models': len(usable_uploaded_models) + 1
        }
        
        with open(os.path.join(MODEL_DIR, 'latest_model.json'), 'w') as f:
            json.dump(model_info, f)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO model_versions 
            (version, path, accuracy, training_data_size, training_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            model_version,
            coreml_path,
            float(accuracy),
            len(X_train),
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"New model version {model_version} created successfully")
        return model_version
    except Exception as e:
        logger.error(f"CoreML conversion failed: {str(e)}")
        return get_current_model_version()

def get_current_model_version():
    MODEL_DIR = os.path.join(os.getenv("RENDER_DISK_PATH", "/var/data"), "models")
    info_path = os.path.join(MODEL_DIR, "latest_model.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        return info.get('version', '1.0.0')
    return '1.0.0'