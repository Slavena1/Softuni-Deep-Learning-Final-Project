"""
data_prep.py
Loading and cleaning functions for the five raw data sources:
Wordbank (child AoA), HSK (adult L2 level), Chinese BabyLM frequencies,
Xu & Li (2020) and Liu et al. (2007) concreteness norms.
"""

import os
import re
import numpy as np
import pandas as pd
from collections import Counter


def load_wordbank(path):
    df_wordbank = pd.read_csv(path, encoding='utf-8', quoting=3)
    df_wordbank.columns = [c.strip('"') for c in df_wordbank.columns]
    df_wordbank['item_definition'] = df_wordbank['item_definition'].str.strip('"')
    df_wordbank['category'] = df_wordbank['category'].str.strip('"')

    # Strip parenthetical disambiguators e.g. 旺旺（狗叫）→ 旺旺
    # These appear in Wordbank but not in any other dataset
    df_wordbank['word'] = df_wordbank['item_definition']\
        .str.replace(r'（[^）]*）', '', regex=True)\
        .str.replace(r'\([^)]*\)', '', regex=True)\
        .str.replace(r'[？！。，、]', '', regex=True)\
        .str.strip()

    # Further cleaning — strip spaces, take first slash-separated option,
    # remove trailing punctuation
    df_wordbank['word'] = df_wordbank['word']\
        .str.replace(' ', '', regex=False)\
        .apply(lambda w: w.split('/')[0] if '/' in w else w)\
        .str.replace(r'[！!？?。，、]', '', regex=True)\
        .str.strip()

    age_cols = [str(a) for a in range(16, 31)]

    def compute_aoa(row, threshold=0.5):
        """Return the first age (months) at which >= 50% of children produce the word."""
        for age in age_cols:
            try:
                if float(row[age]) >= threshold:
                    return int(age)
            except:
                pass
        return np.nan

    df_wordbank['aoa'] = df_wordbank.apply(compute_aoa, axis=1)
    df_aoa = df_wordbank[['word', 'category', 'aoa']].copy()
    return df_aoa


def load_hsk(paths_by_level):
    hsk_dfs = []
    for level, path in paths_by_level.items():
        df = pd.read_csv(path, header=None, names=['word', 'pinyin', 'english'])
        df['hsk_level'] = level
        hsk_dfs.append(df)

    df_hsk = pd.concat(hsk_dfs, ignore_index=True)

    # Resolve duplicates — keep lowest level
    # Sort by hsk_level then word (secondary key) so row order — and therefore
    # any downstream train/test split built on it — is fully deterministic
    # regardless of pandas/numpy version or sort algorithm.
    df_hsk = df_hsk.sort_values(['hsk_level', 'word']).drop_duplicates(
        subset='word', keep='first').reset_index(drop=True)
    return df_hsk


def load_babylm_frequencies(freq_cache_path, parquet_path=None):
    if os.path.exists(freq_cache_path):
        # Load pre-computed frequencies — no need to reprocess
        df_freq = pd.read_csv(freq_cache_path)
        print(f"Loaded pre-computed frequencies: {len(df_freq):,} unique words")
    else:
        # Compute from scratch — only needed once
        print("Computing frequencies from Chinese BabyLM corpus...")
        import jieba
        df_babylm = pd.read_parquet(parquet_path)
        print(f"Corpus: {len(df_babylm):,} documents")
        print(f"Categories: {df_babylm['category'].value_counts().to_dict()}")

        word_counts = Counter()
        for i, text in enumerate(df_babylm['text']):
            words = jieba.lcut(str(text))
            word_counts.update(words)
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1:,} / {len(df_babylm):,} documents...")

        df_freq = pd.DataFrame({
            'word':      list(word_counts.keys()),
            'frequency': list(word_counts.values())
        })

        # Remove punctuation, whitespace and non-Chinese tokens
        # \u4e00-\u9fff is the Unicode range covering all standard Chinese characters
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        df_freq = df_freq[df_freq['word'].apply(
            lambda w: bool(chinese_pattern.search(str(w)))
        )].copy().reset_index(drop=True)
        print(f"After cleaning: {len(df_freq):,} unique Chinese words")

        # Save for future use
        df_freq.to_csv(freq_cache_path, index=False, encoding='utf-8-sig')
        print(f"Saved to {freq_cache_path}")

    df_freq['log_frequency'] = np.log10(df_freq['frequency'].clip(lower=1))
    return df_freq


def load_xuli_concreteness(path):
    df_xuli = pd.read_excel(path)
    df_xuli = df_xuli.rename(columns={
        'Word':                  'word',
        'Mean of Valid Ratings': 'concreteness_raw'
    })
    df_xuli['word'] = df_xuli['word'].str.strip()

    # Reverse scale: original 1=concrete, 5=abstract → reversed 5=concrete, 1=abstract
    df_xuli['concreteness'] = 6 - df_xuli['concreteness_raw']
    return df_xuli


def load_liu_concreteness(path):
    df_liu = pd.read_csv(path, sep='\t', encoding='utf-8')
    df_liu = df_liu.rename(columns={
        'Word': 'word',
        'CON':  'concreteness_raw'  # adjust to actual column name
    })
    df_liu['word'] = df_liu['word'].str.strip()
    return df_liu


def normalize_concreteness(series):
    """Min-max normalize concreteness scores to 0-1 range."""
    return (series - series.min()) / (series.max() - series.min())


def is_reduplication(word):
    """Return True if word is a two-character reduplication (e.g. 妈妈)."""
    return len(word) == 2 and word[0] == word[1]


def build_dataset(df_target, target_col, df_freq, df_concrete_final, word_col='word'):
    """
    Merge a target dataset (Wordbank or HSK) with all feature sources.
    Left join on target words — all target words retained,
    features added where available.
    """
    df = df_target[[word_col, target_col]].copy()
    if 'category' in df_target.columns:
        df['category'] = df_target['category']

    # Frequency from Chinese BabyLM corpus
    df = df.merge(df_freq[['word', 'frequency', 'log_frequency']],
                  on='word', how='left')

    # Concreteness — combined from Xu & Li, Liu et al., and reduplications
    df = df.merge(df_concrete_final[['word', 'concreteness_norm']],
                  on='word', how='left')
    df = df.rename(columns={'concreteness_norm': 'concreteness'})

    # Word length in characters — derived directly
    df['word_length'] = df['word'].str.len()

    return df
