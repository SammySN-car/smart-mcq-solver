"""
Smart MCQ Solver - Main Entry Point
====================================
Ensemble ML + RAG system for solving Multiple Choice Questions.

Usage:
    python main.py --config config.yaml
    python main.py --data_dir ./data --output_dir ./output
"""
import os
import argparse
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.utils import load_config, set_seed, get_device
from src.data import load_data, clean_data, preprocess_text, create_tfidf_features
from src.models.mcqnet import train_mcqnet
from src.models.electra import train_electra
from src.models.minilm import train_minilm
from src.models.rag import train_qwen_rag
from src.ensemble import ensemble_predictions, create_submission

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Smart MCQ Solver')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to output directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    return parser.parse_args()


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI arguments
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if args.seed:
        config['seed'] = args.seed
    
    # Setup
    set_seed(config['seed'])
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and clean data
    logger.info("Loading and cleaning data...")
    data_dir = config['paths']['data_dir']
    
    raw_train = load_data(data_dir, config['data']['train_file'])
    raw_test = load_data(data_dir, config['data']['test_file'])
    
    train = clean_data(raw_train, is_train=True)
    test = clean_data(raw_test, is_train=False)
    
    options = config['data']['options']
    n_train = len(train)
    n_test = len(test)
    
    logger.info(f"Train size: {n_train}, Test size: {n_test}")
    
    # Model 1: TF-IDF MCQNet
    logger.info("=" * 50)
    logger.info("Training Model 1: TF-IDF MCQNet")
    logger.info("=" * 50)
    
    # Preprocess for TF-IDF
    preprocessed_train = preprocess_text(train, options)
    preprocessed_test = preprocess_text(test, options)
    
    # Create TF-IDF features (need to split train for CV)
    from sklearn.model_selection import train_test_split
    train_split, val_split = train_test_split(
        train, test_size=0.2, random_state=config['seed'], stratify=train['answer']
    )
    
    preprocessed_train_split = preprocess_text(train_split, options)
    preprocessed_val_split = preprocess_text(val_split, options)
    
    X_train, X_val, y_train, X_test = create_tfidf_features(
        preprocessed_train_split,
        preprocessed_val_split,
        train_split,
        preprocessed_test,
        options,
        word_max_features=config['model_1']['word_max_features'],
        word_ngram_range=tuple(config['model_1']['word_ngram_range']),
        char_max_features=config['model_1']['char_max_features'],
        char_ngram_range=tuple(config['model_1']['char_ngram_range'])
    )
    
    y_val = np.array([
        1.0 if ans == opt else 0.0
        for ans in val_split['answer']
        for opt in options
    ], dtype=np.float32)
    
    train_oof_mcqnet, test_mcqnet = train_mcqnet(
        X_train, y_train, X_val, y_val, X_test,
        train_split['answer'].tolist(),
        val_split['answer'].tolist(),
        options,
        config['model_1'],
        device,
        n_folds=config['cv']['n_folds']
    )
    
    # Model 2: ELECTRA
    logger.info("=" * 50)
    logger.info("Training Model 2: ELECTRA")
    logger.info("=" * 50)
    
    train_oof_electra, test_electra = train_electra(
        train, test, options, config['model_2'], device,
        n_folds=config['cv']['n_folds']
    )
    
    # Model 3: MiniLM
    logger.info("=" * 50)
    logger.info("Training Model 3: MiniLM")
    logger.info("=" * 50)
    
    train_oof_minilm, test_minilm = train_minilm(
        train, test, options, config['model_3'], device,
        n_folds=config['model_3']['n_folds']
    )
    
    # Model 4: Qwen RAG
    logger.info("=" * 50)
    logger.info("Training Model 4: Qwen RAG")
    logger.info("=" * 50)
    
    test_rag = train_qwen_rag(
        train, test, options,
        config['model_4'], config['rag'],
        device
    )
    
    # Ensemble
    logger.info("=" * 50)
    logger.info("Creating Ensemble Predictions")
    logger.info("=" * 50)
    
    weights = config['ensemble']['weights']
    
    combined_matrix, predictions = ensemble_predictions(
        [test_mcqnet, test_electra, test_minilm, test_rag],
        weights,
        options
    )
    
    # Create submission
    submission = create_submission(test, predictions, os.path.join(output_dir, 'submission.csv'))
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
