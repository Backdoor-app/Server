import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import coremltools as ct
import os
import json
from datetime import datetime
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
    
    logger.info("Training RandomForest classifier")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_vec, y_train, sample_weight=w_train)
    
    accuracy = model.score(X_test_vec, y_test, sample_weight=w_test)
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    timestamp = int(datetime.now().timestamp())
    model_version = f"1.0.{timestamp}"
    
    MODEL_DIR = os.path.join(os.getenv("RENDER_DISK_PATH", "/var/data"), "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    sklearn_path = os.path.join(MODEL_DIR, f"intent_classifier_{model_version}.joblib")
    joblib.dump((vectorizer, model), sklearn_path)
    logger.info(f"Saved sklearn model to {sklearn_path}")
    
    def predict_intent(text):
        processed_text = preprocess_text(text)
        vec_text = vectorizer.transform([processed_text])
        intent = model.predict(vec_text)[0]
        probabilities = model.predict_proba(vec_text)[0]
        return intent, probabilities
    
    logger.info("Converting to CoreML format")
    try:
        class_labels = model.classes_.tolist()
        coreml_model = ct.convert(
            predict_intent,
            inputs=[ct.TensorType(shape=(1,), dtype=str)],
            outputs=[
                ct.TensorType(name='intent'),
                ct.TensorType(name='probabilities', dtype=np.float32)
            ],
            classifier_config=ct.ClassifierConfig(class_labels),
            minimum_deployment_target=ct.target.iOS15
        )
        
        coreml_path = os.path.join(MODEL_DIR, f"model_{model_version}.mlmodel")
        coreml_model.save(coreml_path)
        logger.info(f"Saved CoreML model to {coreml_path}")
        
        model_info = {
            'version': model_version,
            'path': coreml_path,
            'accuracy': float(accuracy),
            'training_data_size': len(X_train),
            'training_date': datetime.now().isoformat()
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