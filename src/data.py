import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


def load_data(file_path):
    print(f"Loading dataset from: {file_path}")
    return pd.read_csv(file_path)


def clean_text_artifact(text, is_option=False, opt_letter=None):
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    replacements = {
        '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
        '\u2014': '-', '\u2013': '-', '\xa0': ' '
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    if not is_option:
        text = re.sub(
            r'^(?:Question\s*\d*|Q\d*|\d+[.\)]|\(\d+\))[:\s]*',
            '', text, flags=re.IGNORECASE
        )
    else:
        if opt_letter:
            pattern = rf'^(?:Option\s*{opt_letter}|{opt_letter}[.\)]|\({opt_letter}\)|\[{opt_letter}\])[:\s]*'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        text = re.sub(r'^(?:[A-Ea-e][.\)]|\([A-Ea-e]\)|\[[A-Ea-e]\])[:\s]*', '', text)

    text = re.sub(r'\s*\?\s*Select the correct (?:option|answer)\.?$', '?', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\?\s*Choose the best (?:option|answer)\.?$', '?', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\(\s*Choose one\s*\)\s*$', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_data(df, is_train=True):
    df = df.copy()

    df['prompt'] = df['prompt'].apply(lambda x: clean_text_artifact(x, is_option=False))

    for opt in ['A', 'B', 'C', 'D', 'E']:
        df[opt] = df[opt].apply(lambda x: clean_text_artifact(x, is_option=True, opt_letter=opt))

    if is_train:
        initial_len = len(df)
        df = df[df['prompt'].str.len() > 0]
        df = df[df['answer'].isin(['A', 'B', 'C', 'D', 'E'])]
        df = df.drop_duplicates(subset=['prompt', 'A', 'B', 'C', 'D', 'E'])
        print(f"Data Cleaning: Removed {initial_len - len(df)} duplicate/invalid rows. Clean unique train size: {len(df)}")

    return df.reset_index(drop=True)


def preprocess_text(df, options):
    rows = []
    for index, row in df.iterrows():
        for option in options:
            rows.append({'text': f"{row['prompt']} {row[option]}"})
    return pd.DataFrame(rows)


def create_tfidf_features(df_1, df_2, raw_data, df_3, options,
                          word_max_features=50000, word_ngram_range=(1, 2),
                          char_max_features=15000, char_ngram_range=(3, 4)):

    tfidf_word = TfidfVectorizer(max_features=word_max_features, ngram_range=word_ngram_range,
                                 sublinear_tf=True, analyzer='word')
    tfidf_char = TfidfVectorizer(max_features=char_max_features, ngram_range=char_ngram_range,
                                 sublinear_tf=True, analyzer='char_wb')

    tfidf_word.fit(df_1['text'])
    tfidf_char.fit(df_1['text'])

    X_tfidf_train = sp.hstack([tfidf_word.transform(df_1['text']),
                                tfidf_char.transform(df_1['text'])]).toarray()
    X_tfidf_val = sp.hstack([tfidf_word.transform(df_2['text']),
                              tfidf_char.transform(df_2['text'])]).toarray()
    X_tfidf_test = sp.hstack([tfidf_word.transform(df_3['text']),
                               tfidf_char.transform(df_3['text'])]).toarray()

    Y_tfidf_train = []
    for index, row in raw_data.iterrows():
        for option in options:
            if row['answer'] == option:
                Y_tfidf_train.append(1.0)
            else:
                Y_tfidf_train.append(0.0)

    return X_tfidf_train, X_tfidf_val, np.array(Y_tfidf_train, dtype=np.float32), X_tfidf_test
