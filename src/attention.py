"""
attention.py

Attention-weight extraction and visualization (Stretch), directly
following the pattern from Ex.4 Problem 9 (inspect a transformer's
attention heads, visualize them, ask whether "important" tokens get
more attention) - applied here to the acquisition-difficulty
question instead of news classification.

Note: attention weights are a contested explainability signal in
the NLP literature - worth one honest sentence acknowledging this
in the notebook rather than presenting attention patterns as
settled proof of anything.
"""


def get_attention_weights(sentence, word, model, tokenizer):
    """
    Run the pretrained transformer (same one loaded in embeddings.py)
    on a sentence containing `word`, with output_attentions=True,
    and return the attention weights for that word's token(s).

    TODO: decide which layer/head to report.
    """
    raise NotImplementedError


def plot_attention_heatmap(sentence, attention_weights, tokens):
    """
    Visualize attention as a heatmap over the sentence tokens.
    TODO: implement with matplotlib/seaborn.
    """
    raise NotImplementedError


def compare_attention_by_difficulty(word_list, difficulty_scores, model, tokenizer):
    """
    The actual analysis: bucket words into easy/medium/hard, and
    compare some summary attention statistic across buckets.

    TODO: use evaluation.bootstrap_ci() before treating any
    cross-bucket difference as meaningful.
    """
    raise NotImplementedError
