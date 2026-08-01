"""
attention.py

Attention-weight extraction and visualization (Stretch), directly
following the pattern from Ex.4 Problem 9 - applied here to the
acquisition-difficulty question instead of news classification.

Sentences for attention analysis are generated from a simple,
part-of-speech-agnostic template ("this word is X"), not mined from
the corpus - a deliberate scope tradeoff for this Stretch section
(see project notes).

Note: attention weights are a contested explainability signal in
the NLP literature - worth one honest sentence acknowledging this
in the notebook rather than presenting attention patterns as
settled proof of anything.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def make_carrier_sentence(word, template="这个词是{word}。"):
    """
    Fill a simple template with the target word. Treats the word as
    a NAMED/QUOTED item ("this word is X") rather than asking it to
    grammatically integrate into the sentence - robust across
    nouns, verbs, adjectives, and function words alike, at the cost
    of naturalism.
    """
    return template.format(word=word)


def _find_word_token_positions(tokens, word):
    """
    Locate the contiguous span of tokens spelling out `word`
    (stripping '##' continuation markers). Returns [] if not found
    (e.g. absorbed into an [UNK] token - worth checking for).
    """
    clean_tokens = [t.replace('##', '') for t in tokens]
    word_chars = list(word)
    n = len(word_chars)
    for start in range(len(clean_tokens) - n + 1):
        if clean_tokens[start:start + n] == word_chars:
            return list(range(start, start + n))
    return []


def get_attention_weights(sentence, word, model, tokenizer):
    """
    Run the pretrained transformer on `sentence` with
    output_attentions=True. Returns (attentions, tokens,
    word_token_positions) - attentions is a tuple of per-layer
    tensors, each (num_heads, seq_len, seq_len).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attentions = tuple(a.squeeze(0) for a in outputs.attentions)
    word_token_positions = _find_word_token_positions(tokens, word)

    return attentions, tokens, word_token_positions


def word_attention_score(attentions, word_token_positions):
    """
    Average attention weight received by the target word's token(s)
    from all OTHER tokens, averaged across all layers and heads.
    Higher = the rest of the sentence attends to this word more.
    Returns np.nan if the word wasn't found in the tokenization.
    """
    if not word_token_positions:
        return np.nan

    scores = []
    for layer_attn in attentions:
        seq_len = layer_attn.shape[-1]
        mask = torch.ones(seq_len, dtype=torch.bool)
        mask[word_token_positions] = False
        for h in range(layer_attn.shape[0]):
            mat = layer_attn[h]
            for pos in word_token_positions:
                incoming = mat[:, pos]
                if mask.sum() > 0:
                    scores.append(incoming[mask].mean().item())
    return float(np.mean(scores)) if scores else np.nan


def plot_attention_heatmap(sentence, attentions, tokens, layer=-1):
    """Heatmap for one sentence, averaged across heads, at `layer`."""
    layer_attn = attentions[layer].mean(dim=0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(layer_attn, xticklabels=tokens, yticklabels=tokens,
                cmap='viridis', ax=ax, cbar_kws={'label': 'Attention weight'})
    ax.set_title(f'Attention heatmap (layer {layer}, avg over heads)\n"{sentence}"')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def compare_attention_by_difficulty(word_list, difficulty_scores, model, tokenizer,
                                     template="这个词是{word}。"):
    """
    Compute each word's attention score via a generated carrier
    sentence, bucket by difficulty_scores into easy/medium/hard.
    Use evaluation.bootstrap_ci on the resulting buckets before
    treating any difference as meaningful.
    """
    records = []
    for word, score in zip(word_list, difficulty_scores):
        sentence = make_carrier_sentence(word, template)
        attentions, tokens, positions = get_attention_weights(sentence, word, model, tokenizer)
        attn_score = word_attention_score(attentions, positions)
        records.append({'word': word, 'difficulty_score': score, 'attention_score': attn_score})

    df = pd.DataFrame(records)
    df['difficulty_bucket'] = pd.qcut(df['difficulty_score'], q=3, labels=['easy', 'medium', 'hard'])
    return df
