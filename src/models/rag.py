"""
Model 4: QLoRA 4-bit Quantized RAG with Qwen.
Uses Wikipedia FAISS index for retrieval and Qwen2.5-7B for scoring.
"""
import os
import gc
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize
import faiss

from src.utils import clear_vram

logger = logging.getLogger(__name__)


def load_wiki_articles(max_articles: int = 50000) -> List[str]:
    """Load Wikipedia articles from Hugging Face or local files.
    
    Args:
        max_articles: Maximum number of articles to load.
    
    Returns:
        List of article texts.
    """
    articles = []
    
    try:
        from datasets import load_dataset
        logger.info("Loading Wikipedia from Hugging Face Hub...")
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split=f"train[:{max_articles}]",
            trust_remote_code=True
        )
        articles = [str(item['text']) for item in ds if len(str(item.get('text', ''))) > 50]
        logger.info(f"Loaded {len(articles)} Wikipedia articles from Hugging Face")
        return articles
    except Exception as e:
        logger.warning(f"Hugging Face load failed: {e}. Falling back to local files...")
    
    # Fallback: search for local parquet/csv/txt files
    search_roots = ["./data", "../data", "."]
    parquets, csvs, txts = [], [], []
    
    for root_dir in search_roots:
        if not os.path.exists(root_dir):
            continue
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                fp = os.path.join(root, f)
                if f.endswith('.parquet'):
                    parquets.append(fp)
                elif f.endswith('.csv') and 'wiki' in fp.lower():
                    csvs.append(fp)
                elif f.endswith('.txt') and 'wiki' in fp.lower():
                    txts.append(fp)
    
    # Load from parquet files
    if parquets:
        logger.info(f"Found {len(parquets)} parquet files")
        for pf in parquets:
            try:
                df = pd.read_parquet(pf)
                text_col = None
                for candidate in ['text', 'content', 'article', 'body', 'document', 'passage']:
                    if candidate in df.columns:
                        text_col = candidate
                        break
                if text_col is None:
                    str_cols = [c for c in df.columns if df[c].dtype == 'object']
                    if str_cols:
                        text_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
                if text_col:
                    texts = df[text_col].dropna().astype(str).tolist()
                    articles.extend(texts)
                    logger.info(f"Loaded {len(texts)} articles from {os.path.basename(pf)}")
            except Exception as e:
                logger.warning(f"Could not load {pf}: {e}")
            if len(articles) >= max_articles:
                break
    
    # Load from CSV files
    if not articles and csvs:
        for cf in csvs:
            try:
                df = pd.read_csv(cf, nrows=max_articles)
                text_col = None
                for candidate in ['text', 'content', 'article', 'body', 'document']:
                    if candidate in df.columns:
                        text_col = candidate
                        break
                if text_col:
                    texts = df[text_col].dropna().astype(str).tolist()
                    articles.extend(texts)
            except Exception as e:
                logger.warning(f"Could not load {cf}: {e}")
    
    # Load from text files
    if not articles and txts:
        for tf_path in txts[:50]:
            try:
                with open(tf_path, 'r', encoding='utf-8', errors='ignore') as fh:
                    articles.append(fh.read())
            except Exception:
                pass
    
    articles = articles[:max_articles]
    logger.info(f"Total Wikipedia articles loaded: {len(articles)}")
    return articles


def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 40) -> List[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: Input text.
        chunk_size: Number of words per chunk.
        chunk_overlap: Number of overlapping words.
    
    Returns:
        List of text chunks.
    """
    chunks = []
    words = text.split()
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
    
    return chunks


def build_wiki_chunks(articles: List[str], chunk_size: int = 200, chunk_overlap: int = 40) -> List[str]:
    """Build text chunks from Wikipedia articles.
    
    Args:
        articles: List of article texts.
        chunk_size: Number of words per chunk.
        chunk_overlap: Number of overlapping words.
    
    Returns:
        List of text chunks.
    """
    all_chunks = []
    for article in articles:
        article_clean = article.strip()
        if len(article_clean) < 50:
            continue
        article_chunks = chunk_text(article_clean, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(article_chunks)
    
    logger.info(f"Total Wikipedia chunks created: {len(all_chunks)}")
    return all_chunks


class WikiRAGIndex:
    """FAISS-based RAG index for Wikipedia chunks."""
    
    def __init__(self, chunks: List[str], max_features: int = 80000, svd_dim: int = 256, top_k: int = 3):
        self.chunks = chunks
        self.top_k = top_k
        
        logger.info(f"Building TF-IDF matrix over {len(chunks)} chunks...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            analyzer='word',
            dtype=np.float32
        )
        tfidf_matrix = self.vectorizer.fit_transform(chunks)
        
        actual_svd_dim = min(svd_dim, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        logger.info(f"Reducing to {actual_svd_dim}-dim via TruncatedSVD...")
        
        self.svd = TruncatedSVD(n_components=actual_svd_dim, random_state=42)
        dense_matrix = self.svd.fit_transform(tfidf_matrix)
        dense_matrix = sk_normalize(dense_matrix, norm='l2').astype(np.float32)
        
        logger.info(f"Building FAISS IndexFlatIP (dim={actual_svd_dim})...")
        self.index = faiss.IndexFlatIP(actual_svd_dim)
        self.index.add(dense_matrix)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: Input query.
            top_k: Number of results to return.
        
        Returns:
            List of relevant chunks.
        """
        k = top_k or self.top_k
        query_sparse = self.vectorizer.transform([query])
        query_dense = self.svd.transform(query_sparse)
        query_dense = sk_normalize(query_dense, norm='l2').astype(np.float32)
        
        distances, indices = self.index.search(query_dense, k)
        return [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]


def build_local_prompt(context: str, question: str, option: str) -> str:
    """Build a prompt for Qwen scoring.
    
    Args:
        context: Retrieved context.
        question: Original question.
        option: Candidate option.
    
    Returns:
        Formatted prompt string.
    """
    return f"Context: {context}\nQuestion: {question}\nCandidate Option: {option}"


def train_qwen_rag(
    train_df,
    test_df,
    options: List[str],
    model_config: Dict[str, Any],
    rag_config: Dict[str, Any],
    device: torch.device
) -> np.ndarray:
    """Train Qwen RAG model for MCQ scoring.
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        options: List of option letters.
        model_config: Model configuration.
        rag_config: RAG configuration.
        device: Torch device.
    
    Returns:
        Test predictions matrix.
    """
    model_name = model_config.get('name', 'Qwen/Qwen2.5-7B-Instruct')
    max_length = model_config.get('max_length', 512)
    use_qlora = model_config.get('use_qlora', True)
    
    # Build RAG index
    logger.info("Building Wikipedia FAISS RAG Index...")
    wiki_articles = load_wiki_articles(max_articles=rag_config.get('max_articles', 50000))
    
    has_wiki_rag = False
    wiki_rag_index = None
    
    if len(wiki_articles) > 0:
        wiki_chunks = build_wiki_chunks(
            wiki_articles,
            chunk_size=rag_config.get('chunk_size', 200),
            chunk_overlap=rag_config.get('chunk_overlap', 40)
        )
        wiki_rag_index = WikiRAGIndex(
            wiki_chunks,
            max_features=rag_config.get('max_features', 80000),
            svd_dim=rag_config.get('svd_dim', 256),
            top_k=rag_config.get('top_k', 3)
        )
        has_wiki_rag = True
    
    # Load Qwen model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    q_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    q_model = None
    
    if use_qlora and torch.cuda.is_available():
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            base_q_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map="auto"
            )
            
            peft_config = LoraConfig(
                r=model_config.get('lora_r', 8),
                lora_alpha=model_config.get('lora_alpha', 16),
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=model_config.get('lora_dropout', 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            q_model = get_peft_model(base_q_model, peft_config)
            logger.info("Loaded QLoRA 4-Bit NF4 Quantized Qwen with LoRA Adapters")
        except Exception as e:
            logger.warning(f"QLoRA initialization failed: {e}. Falling back to float16...")
            q_model = None
    
    if q_model is None:
        q_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)
        logger.info("Loaded Qwen model in float16")
    
    q_model.eval()
    
    # Generate predictions
    n_test = len(test_df)
    test_rag_matrix = np.zeros((n_test, 5))
    
    for i, row in test_df.iterrows():
        option_texts = [str(row[opt]) for opt in options]
        
        if has_wiki_rag and wiki_rag_index is not None:
            query = f"{row['prompt']} {' '.join(option_texts)}"
            passages = wiki_rag_index.retrieve(query, top_k=rag_config.get('top_k', 3))
            context = " ".join(passages)
        else:
            context = f"{row['prompt']} {' '.join(option_texts)}"
        
        scores = []
        for opt in options:
            formatted_text = build_local_prompt(context, str(row['prompt']), str(row[opt]))
            inputs = q_tokenizer(
                formatted_text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length
            ).to(device)
            
            with torch.no_grad():
                outputs = q_model(**inputs, labels=inputs['input_ids'])
                scores.append(-outputs.loss.item())
        
        test_rag_matrix[i] = np.array(scores)
        
        if (i + 1) % 50 == 0:
            logger.info(f"RAG scored {i+1}/{n_test} questions...")
    
    logger.info("Qwen RAG Scoring Complete")
    
    # Cleanup
    clear_vram({
        'q_model': q_model,
        'q_tokenizer': q_tokenizer
    })
    
    if has_wiki_rag:
        del wiki_rag_index, wiki_chunks
        gc.collect()
    
    return test_rag_matrix
