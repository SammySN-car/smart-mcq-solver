import numpy as np
import pandas as pd
from src.utils import normalize_matrix


def ensemble_predictions(test_tfidf_matrix, test_electra_matrix, test_minilm_matrix,
                         test_rag_matrix, weights, options):
    test_tfidf_norm = normalize_matrix(test_tfidf_matrix)
    test_electra_norm = normalize_matrix(test_electra_matrix)
    test_minilm_norm = normalize_matrix(test_minilm_matrix)
    test_rag_norm = normalize_matrix(test_rag_matrix)

    test_combined = (
        weights[0] * test_electra_norm +
        weights[1] * test_tfidf_norm +
        weights[2] * test_minilm_norm +
        weights[3] * test_rag_norm
    )

    top3_indices = np.argsort(-test_combined, axis=1)[:, :3]
    test_predictions = [" ".join([options[idx] for idx in row]) for row in top3_indices]

    return test_combined, test_predictions


def create_submission(test_df, predictions, output_path='submission.csv'):
    sub = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else np.arange(len(test_df)),
        'prediction': predictions
    })

    sub.to_csv(output_path, index=False)
    print(sub.head(10))
    print(f"Total rows: {len(sub)}")

    return sub
