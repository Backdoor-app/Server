"""
Text preprocessing functionality for the Backdoor AI learning system.

This module handles all text normalization, tokenization, and feature extraction 
required for the intent classification models.
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from typing import List, Set, Union, Optional

logger = logging.getLogger(__name__)

def ensure_nltk_resources() -> None:
    """
    Ensures required NLTK resources are available, downloading them if needed.
    """
    try:
        for resource in ['punkt', 'stopwords', 'wordnet']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
    except Exception as e:
        logger.warning(f"Failed to ensure NLTK resources: {e}")

def get_stop_words(language: str = 'english') -> Set[str]:
    """
    Gets stop words for the specified language.
    
    Args:
        language: Language for stop words (default: 'english')
        
    Returns:
        Set of stop words or empty set if resources unavailable
    """
    try:
        return set(stopwords.words(language))
    except Exception as e:
        logger.warning(f"Failed to load stopwords for {language}: {e}")
        return set()

def preprocess_text(text: str, 
                   remove_stopwords: bool = True, 
                   lemmatize: bool = True, 
                   language: str = 'english') -> str:
    """
    Preprocesses text for machine learning models by:
    - Converting to lowercase
    - Tokenizing
    - Removing stop words (optional)
    - Lemmatizing tokens (optional)
    - Removing non-alphanumeric tokens
    - Joining tokens back into a string
    
    Args:
        text: Input text to process
        remove_stopwords: Whether to remove stop words
        lemmatize: Whether to lemmatize tokens
        language: Language for stop words
        
    Returns:
        Preprocessed text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    try:
        tokens = nltk.word_tokenize(text)
    except Exception as e:
        logger.warning(f"Tokenization failed: {e}. Using basic splitting.")
        tokens = text.split()
    
    # Process tokens
    try:
        # Initialize resources if needed
        stop_words = get_stop_words(language) if remove_stopwords else set()
        lemmatizer = WordNetLemmatizer() if lemmatize else None
        
        # Process each token
        processed_tokens = []
        for token in tokens:
            # Skip non-alphanumeric tokens
            if not token.isalnum():
                continue
                
            # Skip stop words
            if token in stop_words:
                continue
                
            # Lemmatize if enabled
            if lemmatize and lemmatizer:
                token = lemmatizer.lemmatize(token)
                
            processed_tokens.append(token)
            
        return ' '.join(processed_tokens)
        
    except Exception as e:
        logger.warning(f"Advanced preprocessing failed: {e}. Using basic alphanumeric filtering.")
        # Fall back to basic filtering
        return ' '.join([token for token in tokens if token.isalnum()])

def extract_features(texts: List[str], max_features: int = 5000, ngram_range: tuple = (1, 2)):
    """
    Extracts features from preprocessed texts using TF-IDF.
    
    Args:
        texts: List of preprocessed text strings
        max_features: Maximum number of features to extract
        ngram_range: Range of n-grams to consider (min_n, max_n)
        
    Returns:
        TF-IDF vectorizer and transformed feature matrix
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        features = vectorizer.fit_transform(texts)
        return vectorizer, features
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise