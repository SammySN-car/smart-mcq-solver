# Smart MCQ Solver

An ensemble machine learning system for solving Multiple Choice Questions (MCQ) on the Kaggle "Smart MCQ Solver Challenge".

**Best Public Leaderboard Score: MAP@3 = 0.76724**

## Problem

Given a scientific prompt and 5 candidate options (A-E), predict the top-3 most probable correct answers. Evaluated via Mean Average Precision at 3 (MAP@3).

**Dataset:** 2,000 training samples, 500 test samples across physics, biology, chemistry, astrophysics, and immunology.

## Architecture

4 models with equal 25% weights:

| Model | Type | Description |
|-------|------|-------------|
| **MCQNet** | Custom Neural Network | TF-IDF features (65K dims) + 2-layer MLP with BatchNorm |
| **ELECTRA** | Pretrained Transformer | Mean Pooling + Multi-Sample Dropout (5 heads) + LLRD |
| **MiniLM-L12** | Bi-Encoder | [CLS] token classification, 384-dim hidden |
| **Qwen RAG** | Generative LLM + RAG | 4-bit QLoRA with Wikipedia FAISS index |

### Model Details

**1. MCQNet (25%)**
- Dual n-gram features: word (1,2) 50K + char (3,4) 15K = 65K dims
- 2-layer MLP with BatchNorm and Dropout

**2. ELECTRA-base (25%)**
- 5-Fold Stratified Cross-Validation
- Mean Pooling + Multi-Sample Dropout (5 heads)
- Layer-wise Learning Rate Decay (LLRD, gamma=0.9)

**3. MiniLM-L12 Bi-Encoder (25%)**
- 384-dimensional hidden representation
- [CLS] token classification head

**4. Qwen RAG (25%)**
- Wikipedia FAISS index for retrieval
- 4-bit NF4 quantization with LoRA adapters

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
- **RAG** with Wikipedia provides retrieval-augmented context

## License

MIT License
