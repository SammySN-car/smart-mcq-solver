import os
import gc
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize
import faiss
from src.utils import clear_vram


def find_wiki_files():
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

    return parquets, csvs, txts


def load_wiki_articles(max_articles=50000):
    articles = []

    try:
        from datasets import load_dataset
        print("Attempting direct load from Hugging Face Hub (wikimedia/wikipedia)...")
        ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                          split=f"train[:{max_articles}]", trust_remote_code=True)
        articles = [str(item['text']) for item in ds if len(str(item.get('text', ''))) > 50]
        print(f"Successfully loaded {len(articles)} raw Wikipedia articles directly from Hugging Face!")
        return articles
    except Exception as e:
        print(f"Hugging Face direct load unavailable/offline ({e}). Falling back to local files...")

    parquets, csvs, txts = find_wiki_files()

    if parquets:
        print(f"Found {len(parquets)} parquet file(s). Loading Wikipedia articles...")
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
                    print(f"  Loaded {len(texts)} articles from {os.path.basename(pf)} (col: {text_col})")
            except Exception as e:
                print(f"  Warning: Could not load {pf}: {e}")
            if len(articles) >= max_articles:
                break

    if not articles and csvs:
        print(f"Found {len(csvs)} wiki CSV file(s). Loading...")
        for cf in csvs:
            try:
                df = pd.read_csv(cf, nrows=max_articles)
                text_col = None
                for candidate in ['text', 'content', 'article', 'body', 'document']:
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
                    print(f"  Loaded {len(texts)} articles from {os.path.basename(cf)}")
            except Exception as e:
                print(f"  Warning: Could not load {cf}: {e}")

    if not articles and txts:
        print(f"Found {len(txts)} wiki text file(s). Loading...")
        for tf_path in txts[:50]:
            try:
                with open(tf_path, 'r', encoding='utf-8', errors='ignore') as fh:
                    articles.append(fh.read())
            except Exception:
                pass

    articles = articles[:max_articles]
    print(f"Total Wikipedia articles loaded: {len(articles)}")
    return articles


def chunk_text(text, chunk_size=200, chunk_overlap=40):
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


def build_wiki_chunks(articles, chunk_size=200, chunk_overlap=40):
    all_chunks = []

    for article in articles:
        article_clean = article.strip()
        if len(article_clean) < 50:
            continue
        article_chunks = chunk_text(article_clean, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(article_chunks)

    print(f"Total Wikipedia chunks created: {len(all_chunks)}")
    return all_chunks


class WikiRAGIndex:
    def __init__(self, chunks, max_features=80000, svd_dim=256, top_k=3):
        self.chunks = chunks
        self.top_k = top_k

        print(f"Building TF-IDF matrix over {len(chunks)} chunks (max_features={max_features})...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2),
            sublinear_tf=True, analyzer='word', dtype=np.float32
        )
        tfidf_matrix = self.vectorizer.fit_transform(chunks)
        print(f"  TF-IDF sparse matrix shape: {tfidf_matrix.shape}")

        actual_svd_dim = min(svd_dim, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        print(f"  Reducing to {actual_svd_dim}-dim dense vectors via TruncatedSVD...")

        self.svd = TruncatedSVD(n_components=actual_svd_dim, random_state=42)
        dense_matrix = self.svd.fit_transform(tfidf_matrix)
        dense_matrix = sk_normalize(dense_matrix, norm='l2').astype(np.float32)

        print(f"  Building FAISS IndexFlatIP (dim={actual_svd_dim})...")
        self.index = faiss.IndexFlatIP(actual_svd_dim)
        self.index.add(dense_matrix)
        print(f"  FAISS index built with {self.index.ntotal} vectors!")

    def retrieve(self, query, top_k=None):
        k = top_k or self.top_k
        query_sparse = self.vectorizer.transform([query])
        query_dense = self.svd.transform(query_sparse)
        query_dense = sk_normalize(query_dense, norm='l2').astype(np.float32)
        distances, indices = self.index.search(query_dense, k)
        return [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]


def build_local_prompt(context, main_q, option):
    return f"Context: {context}\nQuestion: {main_q}\nCandidate Option: {option}"


def train_qwen_rag(train, test, options, model_name, model_config, rag_config, device):
    max_articles = rag_config.get('max_articles', 50000)
    chunk_size = rag_config.get('chunk_size', 200)
    chunk_overlap = rag_config.get('chunk_overlap', 40)
    max_features = rag_config.get('max_features', 80000)
    svd_dim = rag_config.get('svd_dim', 256)
    top_k = rag_config.get('top_k', 3)

    print("--- Building Wikipedia FAISS RAG Index from Raw Dump ---")
    wiki_articles = load_wiki_articles(max_articles=max_articles)

    HAS_WIKI_RAG = False
    wiki_rag_index = None

    if len(wiki_articles) > 0:
        wiki_chunks = build_wiki_chunks(wiki_articles, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        wiki_rag_index = WikiRAGIndex(wiki_chunks, max_features=max_features, svd_dim=svd_dim, top_k=top_k)
        HAS_WIKI_RAG = True
    else:
        print("WARNING: No Wikipedia dump found. Falling back to question-only context.")

    def retrieve_context(prompt, option_texts, top_k=top_k):
        if HAS_WIKI_RAG:
            query = f"{prompt} {' '.join(option_texts)}"
            passages = wiki_rag_index.retrieve(query, top_k=top_k)
            return " ".join(passages)
        else:
            return f"{prompt} {' '.join(option_texts)}"

    del wiki_articles
    gc.collect()
    print("Wikipedia FAISS RAG Index ready for retrieval!")

    print("--- Running QLoRA 4-Bit Adapter-Tuned Wikipedia RAG Scoring with Qwen2.5-7B-Instruct ---")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import BitsAndBytesConfig
        has_peft = True
    except ImportError:
        has_peft = False

    q_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    q_model = None

    if has_peft and torch.cuda.is_available():
        try:
            import bitsandbytes
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
            )
            base_q_model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config,
                trust_remote_code=True, device_map="auto"
            )
            peft_config = LoraConfig(
                r=8, lora_alpha=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
            )
            q_model = get_peft_model(base_q_model, peft_config)
            print("Loaded QLoRA 4-Bit NF4 Quantized Qwen2.5-7B-Instruct with LoRA Adapters!")
        except Exception as e:
            print(f"QLoRA 4-bit initialization note: {e}")
            print("Falling back to float16 model loading...")
            q_model = None

    if q_model is None:
        q_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device)
        print("Loaded Local Qwen2.5-7B-Instruct for Generative RAG Scoring!")

    q_model.eval()

    n_test = len(test)
    test_rag_matrix = np.zeros((n_test, 5))

    for i, row in test.iterrows():
        option_texts = [str(row[opt]) for opt in options]
        context = retrieve_context(str(row['prompt']), option_texts, top_k=top_k)

        scores = []
        for opt_idx, opt in enumerate(options):
            formatted_text = build_local_prompt(context, str(row['prompt']), str(row[opt]))
            inputs = q_tokenizer(formatted_text, return_tensors='pt', truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = q_model(**inputs, labels=inputs['input_ids'])
                scores.append(-outputs.loss.item())

        test_rag_matrix[i] = np.array(scores)

        if (i + 1) % 50 == 0:
            print(f"  RAG scored {i+1}/{n_test} questions...")

    print("QLoRA 7B Wikipedia RAG Scoring Complete!")

    clear_vram(['q_model', 'q_tokenizer'])

    if HAS_WIKI_RAG:
        del wiki_rag_index, wiki_chunks
        gc.collect()
        print("Wikipedia index freed from memory.")

    return test_rag_matrix


