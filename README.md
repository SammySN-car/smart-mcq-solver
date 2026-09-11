# Smart MCQ Solver

An ensemble machine learning system for solving 5-option Multiple Choice Questions (MCQ) on scientific topics.

**Best Public Leaderboard Score: MAP@3 = 0.76724**

## Competition Overview

The **Smart MCQ Solver Challenge** is a Kaggle competition that challenges participants to predict the top-3 most probable correct answers for scientific multiple-choice questions.

### Problem Details

- **Task:** Given a scientific prompt and 5 candidate options (A-E), predict the top-3 most probable correct answers
- **Evaluation Metric:** Mean Average Precision at 3 (MAP@3)
- **Dataset Size:** 2,000 training samples, 500 test samples
- **Scientific Domains:** Physics, biology, chemistry, astrophysics, immunology
- **Question Format:** 5 options per question, single correct answer

### MAP@3 Metric

MAP@3 measures the average precision of the top-3 predicted answers across all questions. It rewards models that place the correct answer higher in their top-3 predictions. For example:
- If correct answer is ranked 1st: Precision@3 = 1.00
- If correct answer is ranked 2nd: Precision@3 = 0.50
- If correct answer is ranked 3rd: Precision@3 = 0.33

## Architecture

4 models with equal 25% weights:

| Model | Type | Description |
|-------|------|-------------|
| **MCQNet** | Custom Neural Network | TF-IDF features (65K dims) + 3-layer MLP with BatchNorm |
| **ELECTRA** | Pretrained Transformer | Mean Pooling + Multi-Sample Dropout (5 heads) + LLRD |
| **MiniLM-L12** | Bi-Encoder | [CLS] token classification, 384-dim hidden |
| **Qwen RAG** | Generative LLM + RAG | 4-bit QLoRA with Wikipedia FAISS index |

### Model Details

**1. MCQNet (25%)**
- Dual n-gram features: word (1,2) 50K + char (3,4) 15K = 65K dims
- 3-layer MLP: input_dim → 256 → 64 → 1 (with BatchNorm + Dropout)
- 5-Fold Stratified Cross-Validation
- Linear warmup scheduler

**2. ELECTRA-base (25%)**
- 5-Fold Stratified Cross-Validation
- Mean Pooling + Multi-Sample Dropout (5 heads, p=0.2)
- Layer-wise Learning Rate Decay (LLRD, gamma=0.9)
- Gradient accumulation (4 steps) for effective batch size 32

**3. MiniLM-L12 Bi-Encoder (25%)**
- 384-dimensional hidden representation
- [CLS] token classification head
- 3-Fold Cross-Validation
- Sequence length: 128 tokens

**4. Qwen RAG (25%)**
- Wikipedia FAISS index for retrieval (50K articles, chunked)
- TF-IDF → TruncatedSVD (256-dim) → FAISS IndexFlatIP
- 4-bit NF4 quantization with LoRA adapters (r=8, alpha=16)
- Generative scoring via next-token prediction loss

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

### Required Data Files

Place these in your data directory:
- 	rain.csv - Training data with columns: id, prompt, A, B, C, D, E, answer
- 	est.csv - Test data with columns: id, prompt, A, B, C, D, E

## Key Findings

- **LLRD** prevented catastrophic forgetting during fine-tuning
- **Multi-Sample Dropout** provided +0.008 MAP@3 improvement
- **Gradient Accumulation** enabled training with batch_size=8 on limited GPU
- **RAG** with Wikipedia provides retrieval-augmented context for scientific questions
- **Ensemble diversity** across TF-IDF, transformer, and generative models improves robustness

## License

MIT License
