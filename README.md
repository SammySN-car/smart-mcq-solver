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

MAP@3 measures the average precision of the top-3 predicted answers across all questions.

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
- 3-layer MLP: input_dim -> 256 -> 64 -> 1

**2. ELECTRA-base (25%)**
- 5-Fold Stratified Cross-Validation
- Mean Pooling + Multi-Sample Dropout (5 heads, p=0.2)
- LLRD (gamma=0.9)

**3. MiniLM-L12 Bi-Encoder (25%)**
- 384-dimensional hidden representation
- [CLS] token classification head

**4. Qwen RAG (25%)**
- Wikipedia FAISS index for retrieval (50K articles, chunked)
- 4-bit NF4 quantization with LoRA adapters

## Project Structure

    smart_mcq_solver/
    |-- config.yaml
    |-- requirements.txt
    |-- main.py
    |-- src/
        |-- utils.py
        |-- data.py
        |-- ensemble.py
        |-- models/
            |-- mcqnet.py
            |-- electra.py
            |-- minilm.py
            |-- rag.py

## Installation

```bash
git clone https://github.com/SammySN-car/smart-mcq-solver.git
cd smart-mcq-solver
pip install -r requirements.txt
```

## Usage

```bash
python main.py --data_dir ./data --output_dir ./output
```

### Required Data Files

- train.csv - Training data with columns: id, prompt, A, B, C, D, E, answer
- test.csv - Test data with columns: id, prompt, A, B, C, D, E

## Key Findings

- **LLRD** prevented catastrophic forgetting during fine-tuning
- **Multi-Sample Dropout** provided +0.008 MAP@3 improvement
- **Gradient Accumulation** enabled training with batch_size=8 on limited GPU
- **RAG** with Wikipedia provides retrieval-augmented context

## License

MIT License