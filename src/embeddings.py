"""
embeddings.py

Transfer learning component: extract word embeddings from a pretrained
Chinese transformer (frozen weights), for use as features in
neural_models.py.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

_MODEL_CACHE = {}


def load_pretrained_model(model_name="hfl/chinese-macbert-base"):
    """
    Load a pretrained Chinese transformer + tokenizer, frozen
    (no fine-tuning - used purely as a fixed feature extractor).
    Cached in-memory by model_name so repeated calls don't re-load.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    _MODEL_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer


def get_word_embedding(word, model, tokenizer, pooling="mean"):
    """
    Get a single embedding vector for one Mandarin word, in isolation.

    pooling:
      'mean' - average sub-token hidden states, excluding [CLS]/[SEP]
      'cls'  - use the [CLS] token's hidden state
    """
    device = next(model.parameters()).device
    inputs = tokenizer(word, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state.squeeze(0)

    if pooling == "cls":
        vec = hidden[0]
    else:
        # mean pool over the actual word tokens (exclude [CLS] and [SEP])
        vec = hidden[1:-1].mean(dim=0) if hidden.shape[0] > 2 else hidden.mean(dim=0)

    return vec.cpu().numpy()


def build_embedding_matrix(word_list, model, tokenizer, pooling="mean", verbose=True):
    """
    Batch version of get_word_embedding. Returns (n_words, dim), in
    the same row order as word_list - important: this positional
    order is what lets the result be sliced with the same
    idx_train/idx_test used for the baseline models.
    """
    embeddings = []
    for i, word in enumerate(word_list):
        if verbose and i % 100 == 0:
            print(f"  {i}/{len(word_list)} words embedded...")
        embeddings.append(get_word_embedding(word, model, tokenizer, pooling=pooling))

    if verbose:
        print(f"  {len(word_list)}/{len(word_list)} done.")

    return np.vstack(embeddings)
