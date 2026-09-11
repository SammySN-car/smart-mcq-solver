# Smart MCQ Solver

An ensemble machine learning and Retrieval-Augmented Generation (RAG) system for solving Multiple Choice Questions (MCQ).

## Architecture

This system combines 4 models with equal 25% weights:

| Model | Description | Weight |
|-------|-------------|--------|
| **MCQNet** | Custom TF-IDF Neural Network with BatchNorm and Dropout | 25% |
| **ELECTRA** | Primary Contextual Discriminator with LLRD and Multi-Sample Dropout | 25% |
| **MiniLM** | Compact Bi-Encoder using [CLS] token classification | 25% |
| **Qwen RAG** | QLoRA 4-Bit Quantized RAG with Wikipedia FAISS Index | 25% |

## Project Structure

`
smart_mcq_solver/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── main.py                     # Main entry point
├── src/
│   ├── __init__.py
│   ├── utils.py                # Utility functions
│   ├── data.py                 # Data loading and preprocessing
│   ├── ensemble.py             # Ensemble logic
│   └── models/
│       ├── __init__.py
│       ├── mcqnet.py           # Model 1: TF-IDF MCQNet
│       ├── electra.py          # Model 2: ELECTRA
│       ├── minilm.py           # Model 3: MiniLM
│       └── rag.py              # Model 4: Qwen RAG
├── data/                       # Data directory (not in git)
├── models_saved/               # Saved model weights (not in git)
├── notebooks/                  # Jupyter notebooks
├── reports/                    # PDF reports
└── streamlit_deployment/       # Streamlit web app
`

## Installation

1. Clone the repository:
`ash
git clone <repository-url>
cd smart_mcq_solver
`

2. Create a virtual environment:
`ash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
`

3. Install dependencies:
`ash
pip install -r requirements.txt
`

## Usage

### Training

`ash
python main.py --config config.yaml
`

With custom paths:
`ash
python main.py --data_dir /path/to/data --output_dir /path/to/output
`

### Configuration

Edit config.yaml to customize:
- Model hyperparameters
- Cross-validation settings
- RAG configuration
- Ensemble weights

## Data

Place your data files in the data/ directory:
- 	rain.csv - Training data with columns: id, prompt, A, B, C, D, E, answer
- 	est.csv - Test data with columns: id, prompt, A, B, C, D, E

## Streamlit App

To run the Streamlit web app:
`ash
cd streamlit_deployment
pip install -r requirements.txt
streamlit run app.py
`

## Results

The ensemble achieves strong performance on the MCQ solving task by leveraging:
- **TF-IDF features** for keyword matching
- **ELECTRA** for deep contextual understanding
- **MiniLM** for efficient sentence embeddings
- **RAG** for retrieval-augmented generation with Wikipedia

## License

MIT License
