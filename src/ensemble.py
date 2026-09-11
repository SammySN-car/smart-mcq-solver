"""
Ensemble logic for combining model predictions.
"""
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from src.utils import normalize_matrix

logger = logging.getLogger(__name__)


def ensemble_predictions(
    predictions: List[np.ndarray],
    weights: List[float],
    options: List[str],
    top_k: int = 3
) -> Tuple[np.ndarray, List[str]]:
    """Combine predictions from multiple models using weighted average.
    
    Args:
        predictions: List of prediction matrices (n_samples x 5).
        weights: List of weights for each model.
        options: List of option letters.
        top_k: Number of top predictions to return.
    
    Returns:
        Tuple of (combined_matrix, predictions_list).
    """
    if len(predictions) != len(weights):
        raise ValueError("Number of predictions must match number of weights")
    
    # Normalize each prediction matrix
    normalized = [normalize_matrix(pred) for pred in predictions]
    
    # Weighted average
    combined = np.zeros_like(normalized[0])
    for pred, weight in zip(normalized, weights):
        combined += weight * pred
    
    # Get top-k predictions
    top3_indices = np.argsort(-combined, axis=1)[:, :top_k]
    predictions_list = [" ".join([options[idx] for idx in row]) for row in top3_indices]
    
    return combined, predictions_list


def create_submission(
    test_df: pd.DataFrame,
    predictions: List[str],
    output_path: str = "submission.csv"
) -> pd.DataFrame:
    """Create submission DataFrame and save to CSV.
    
    Args:
        test_df: Test DataFrame.
        predictions: List of prediction strings.
        output_path: Path to save the CSV file.
    
    Returns:
        Submission DataFrame.
    """
    submission = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else np.arange(len(test_df)),
        'prediction': predictions
    })
    
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Total rows: {len(submission)}")
    logger.info(submission.head(10))
    
    return submission
