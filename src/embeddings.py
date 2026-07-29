"""
embeddings.py

Transfer learning component: extract word embeddings from a pretrained
Chinese transformer (frozen weights), for use as features in
neural_models.py - the same "frozen base + custom head" pattern taught
in the Vision Models exercise (Ex.6), applied to text.

Model choice TODO: pick a pretrained Chinese encoder, e.g.
'hfl/chinese-macbert-base' (recommended) or 'bert-base-chinese' as a
simpler fallback.
"""

import numpy as np


def load_pretrained_model(model_name):
    """
    Load a pretrained Chinese transformer + tokenizer, frozen
    (no fine-tuning - used purely as a fixed feature extractor).

    TODO: implement with transformers.AutoModel / AutoTokenizer.
    Freeze all parameters (requires_grad = False) - this is the
    explicit "frozen base" transfer-learning pattern from Ex.6, and
    matches the Language Models lecture's own rule of thumb ("more
    training data = less frozen layers") given our small word list.
    """
    raise NotImplementedError


def get_word_embedding(word, model, tokenizer, pooling="mean"):
    """
    Get a single embedding vector for one Mandarin word.

    TODO: tokenize word (consider whether to embed the word in
    isolation or in a carrier sentence - isolation is simpler but
    a carrier sentence may give more meaningful contextual
    embeddings; worth trying both and comparing).

    pooling: how to combine sub-token embeddings into one vector
    ('mean' pooling over tokens is a reasonable default; '[CLS]'
    token is another common option - worth documenting which was
    chosen and why in the notebook).
    """
    raise NotImplementedError


def build_embedding_matrix(word_list, model, tokenizer):
    """
    Batch version of get_word_embedding for the full word list.
    Returns a (n_words, embedding_dim) numpy array, in the same
    row order as word_list, so it lines up with build_dataset()'s
    output from data_prep.py.

    TODO: implement, with progress logging since this could be slow
    over the full word list - consider caching to disk (data/) so
    this only needs to run once, same caching principle as the
    LLM API responses.
    """
    raise NotImplementedError
