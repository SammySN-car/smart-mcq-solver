"""
Data loading and preprocessing functions.
"""
import os
import re
import logging
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def load_data(data_dir: str, filename: str) -> pd.DataFrame:
    """Load CSV data from the data directory.
    
    Args:
        data_dir: Path to the data directory.
        filename: Name of the CSV file.
    
    Returns:
        Loaded DataFrame.
    """
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    logger.info(f"Loading dataset from: {file_path}")
    return pd.read_csv(file_path)


def clean_text_artifact(text: str, is_option: bool = False, opt_letter: str = None) -> str:
    """Clean text artifacts and normalize.
    
    Args:
        text: Input text to clean.
        is_option: Whether the text is an option (A-E).
        opt_letter: Option letter (A-E) if applicable.
    
    Returns:
        Cleaned text.
    """
    if not isinstance(text, str):
        text = str(text)
    
    text = text.strip()
    
    # Replace special characters
    replacements = {
        '"': '"', '"': '"', "'": "'", "'": "'",
        '—': '-', '–': '-', '\\xa0': ' '
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Clean question prefixes
    if not is_option:
        text = re.sub(
            r'^(?:Question\s*\d*|Q\d*|\d+[.\)]|\(\d+\))[:\s]*',
            '', text, flags=re.IGNORECASE
        )
    else:
        if opt_letter:
            pattern = rf'^(?:Option\s*{opt_letter}|{opt_letter}[.\)]|\({opt_letter}\)|\[{opt_letter}\])[:\s]*'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(?:[A-Ea-e][.\)]|\([A-Ea-e]\)|\[[A-Ea-e]\])[:\s]*', '', text)
    
    # Remove trailing instructions
    text = re.sub(r'\s*\?\s*Select the correct (?:option|answer)\.?$', '?', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\?\s*Choose the best (?:option|answer)\.?$', '?', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\(\s*Choose one\s*\)\s*$', '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Clean the dataset by removing invalid rows and duplicates.
    
    Args:
        df: Input DataFrame.
        is_train: Whether this is training data.
    
    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    
    # Clean prompt
    df['prompt'] = df['prompt'].apply(lambda x: clean_text_artifact(x, is_option=False))
    
    # Clean options
    for opt in ['A', 'B', 'C', 'D', 'E']:
        df[opt] = df[opt].apply(lambda x: clean_text_artifact(x, is_option=True, opt_letter=opt))
    
    if is_train:
        initial_len = len(df)
        df = df[df['prompt'].str.len() > 0]
        df = df[df['answer'].isin(['A', 'B', 'C', 'D', 'E'])]
        df = df.drop_duplicates(subset=['prompt', 'A', 'B', 'C', 'D', 'E'])
        removed = initial_len - len(df)
        logger.info(f"Data Cleaning: Removed {removed} duplicate/invalid rows. Clean size: {len(df)}")
    
    return df.reset_index(drop=True)


def preprocess_text(df: pd.DataFrame, options: List[str]) -> pd.DataFrame:
    """Create combined text features for TF-IDF.
    
    Args:
        df: Input DataFrame.
        options: List of option letters.
    
    Returns:
        DataFrame with 'text' column.
    """
    rows = []
    for _, row in df.iterrows():
        for option in options:
            rows.append({'text': f"{row['prompt']} {row[option]}"})
    return pd.DataFrame(rows)


def create_tfidf_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    raw_data: pd.DataFrame,
    df_test: pd.DataFrame,
    options: List[str],
    word_max_features: int = 50000,
    word_ngram_range: Tuple[int, int] = (1, 2),
    char_max_features: int = 15000,
    char_ngram_range: Tuple[int, int] = (3, 4)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create TF-IDF features combining word and character n-grams.
    
    Args:
        df_train: Training data with 'text' column.
        df_val: Validation data with 'text' column.
        raw_data: Raw training data with 'answer' column.
        df_test: Test data with 'text' column.
        options: List of option letters.
        word_max_features: Max features for word TF-IDF.
        word_ngram_range: N-gram range for word TF-IDF.
        char_max_features: Max features for char TF-IDF.
        char_ngram_range: N-gram range for char TF-IDF.
    
    Returns:
        Tuple of (X_train, X_val, y_train, X_test).
    """
    # Word-level TF-IDF
    tfidf_word = TfidfVectorizer(
        max_features=word_max_features,
        ngram_range=word_ngram_range,
        sublinear_tf=True,
        analyzer='word'
    )
    
    # Character-level TF-IDF
    tfidf_char = TfidfVectorizer(
        max_features=char_max_features,
        ngram_range=char_ngram_range,
        sublinear_tf=True,
        analyzer='char_wb'
    )
    
    # Fit on training text
    tfidf_word.fit(df_train['text'])
    tfidf_char.fit(df_train['text'])
    
    # Transform
    X_train = sp.hstack([
        tfidf_word.transform(df_train['text']),
        tfidf_char.transform(df_train['text'])
    ]).toarray()
    
    X_val = sp.hstack([
        tfidf_word.transform(df_val['text']),
        tfidf_char.transform(df_val['text'])
    ]).toarray()
    
    X_test = sp.hstack([
        tfidf_word.transform(df_test['text']),
        tfidf_char.transform(df_test['text'])
    ]).toarray()
    
    # Create labels
    Y_train = []
    for _, row in raw_data.iterrows():
        for option in options:
            Y_train.append(1.0 if row['answer'] == option else 0.0)
    
    return X_train, X_val, np.array(Y_train, dtype=np.float32), X_test
