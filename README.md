# Smart MCQ Solver

An ensemble machine learning system for solving Multiple Choice Questions (MCQ) on the Kaggle "Smart MCQ Solver Challenge".

**Best Public Leaderboard Score: MAP@3 = 0.76724**

## Problem

Given a scientific prompt and 5 candidate options (A-E), predict the top-3 most probable correct answers. Evaluated via Mean Average Precision at 3 (MAP@3).

**Dataset:** 2,000 training samples, 500 test samples across physics, biology, chemistry, astrophysics, and immunology.

## Architecture

| Model | Type | MAP@3 | Weight |
|-------|------|-------|--------|
| **ELECTRA-base** | Pretrained Transformer | ~0.76 | 44.1% |
| **TF-IDF NN** | Custom Neural Network | ~0.72 | 44.1% |
| **MiniLM-L12** | Bi-Encoder | ~0.73 | 9.8% |
| **Length Prior** | EDA-based Bias | - | 2.0% |

### Model Details

**1. ELECTRA-base (44.1%)**
- 5-Fold Stratified Cross-Validation
- Mean Pooling + Multi-Sample Dropout (5 heads)
- Layer-wise Learning Rate Decay (LLRD, gamma=0.9)

**2. TF-IDF Neural Network (44.1%)**
- Dual n-gram features: word (1,2) 50K + char (3,4) 15K = 65K dims
- 2-layer MLP with BatchNorm and Dropout

**3. MiniLM-L12 Bi-Encoder (9.8%)**
- 384-dimensional hidden representation
- [CLS] token classification head
- Architectural diversity for ensemble

**4. Option Length Prior (2.0%)**
- Discovered via EDA: 40.55% of correct answers are longest option

## Project Structure

`
smart_mcq_solver/
├── config.yaml              # All hyperparameters
├── requirements.txt         # Dependencies
├── main.py                  # Entry point
└── src/
    ├── utils.py             # set_seed, mapk, normalize_matrix
    ├── data.py              # load_data, clean_data, create_tfidf_features
    ├── ensemble.py          # ensemble_predictions, create_submission
    └── models/
        ├── mcqnet.py        # TF-IDF Neural Network
        ├── electra.py       # ELECTRA + LLRD + Multi-Sample Dropout
        ├── minilm.py        # MiniLM Bi-Encoder
        └── rag.py           # Wikipedia RAG with Qwen
`

## Installation

`ash
git clone https://github.com/SammySN-car/smart-mcq-solver.git
cd smart-mcq-solver
pip install -r requirements.txt
`

## Usage

`ash
python main.py --data_dir ./data --output_dir ./output
`

## Key Findings

- **LLRD** prevented catastrophic forgetting during fine-tuning
- **Multi-Sample Dropout** provided +0.008 MAP@3 improvement
- **Gradient Accumulation** enabled training with batch_size=8 on limited GPU
- **Option Length Bias** (40.55%) was the most impactful EDA discovery

## License

MIT License
