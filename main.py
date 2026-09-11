import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch

from src.utils import load_config, set_seed, get_device
from src.data import load_data, clean_data
from src.models.mcqnet import train_mcqnet
from src.models.electra import train_electra
from src.models.minilm import train_minilm
from src.models.rag import train_qwen_rag
from src.ensemble import ensemble_predictions, create_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = args.data_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    SEED = config['seed']
    options = config['data']['options']
    device = get_device()

    set_seed(SEED)

    print("Loading data...")
    raw_train = load_data(os.path.join(data_dir, 'train.csv'))
    raw_test = load_data(os.path.join(data_dir, 'test.csv'))

    train = clean_data(raw_train, is_train=True)
    test = clean_data(raw_test, is_train=False)

    n_train, n_test = len(train), len(test)
    print(f"Train size: {n_train}, Test size: {n_test}")

    model_name_1 = config['model_1']['model_name']
    model_name_2 = config['model_2']['model_name']
    model_name_3 = config['model_3']['model_name']
    model_name_4 = config['model_4']['model_name']

    print("\n" + "="*50)
    print("Training Model 1: TF-IDF MCQNet")
    print("="*50)
    train_tfidf_oof, test_tfidf_matrix = train_mcqnet(
        train, test, options, config['model_1'], device, data_dir,
        n_folds=config['cv']['n_folds']
    )

    print("\n" + "="*50)
    print("Training Model 2: ELECTRA")
    print("="*50)
    train_electra_oof, test_electra_matrix = train_electra(
        train, test, options, model_name_2, config['model_2'], device,
        n_folds=config['cv']['n_folds']
    )

    print("\n" + "="*50)
    print("Training Model 3: MiniLM")
    print("="*50)
    train_minilm_oof, test_minilm_matrix = train_minilm(
        train, test, options, model_name_3, config['model_3'], device,
        n_folds=config['model_3']['n_folds']
    )

    print("\n" + "="*50)
    print("Training Model 4: Qwen RAG")
    print("="*50)
    test_rag_matrix = train_qwen_rag(
        train, test, options, model_name_4,
        config['model_4'], config['rag'], device
    )

    print("\n" + "="*50)
    print("Creating Ensemble")
    print("="*50)
    weights = config['ensemble']['weights']
    test_combined, predictions = ensemble_predictions(
        test_tfidf_matrix, test_electra_matrix, test_minilm_matrix,
        test_rag_matrix, weights, options
    )

    submission = create_submission(test, predictions, os.path.join(output_dir, 'submission.csv'))
    print("\nDone!")


if __name__ == '__main__':
    main()
