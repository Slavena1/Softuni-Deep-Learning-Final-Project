"""
evaluation.py

Two-tier "does the model know this word?" methodology:

  Tier 1 - local model (full internal access): log-likelihood,
  perplexity, masked-word prediction. The classic BabyLM-style
  AoA-prediction method.

  Tier 2 - API models, all three providers (no logprobs available
  for any of them - verified for both Claude and Qwen/DashScope):
  sentence completion and minimal-pair judgments, scored via
  LLM-as-judge (see Language Models exercise, Problem 9), not
  exact/semantic string match.

Also home to the bootstrap/uncertainty utilities - direct lesson
from the ML project's 86% feedback: don't compare importances or
scores across models/datasets without an uncertainty estimate.
Used in: Section 5 (architecture comparison), Section 6
(uncertainty analysis proper), Section 7 (stratified error
analysis), and Section 4.4 if built (attention comparison).
"""

import numpy as np


# ---- Tier 1: local model, internal access ----

def masked_word_probability(sentence, target_word, model, tokenizer):
    """TODO: implement with a masked-LM model/head."""
    raise NotImplementedError


def sentence_log_likelihood(sentence, model, tokenizer):
    """TODO: implement, document which conditioning was used."""
    raise NotImplementedError


def perplexity(sentences, model, tokenizer):
    """TODO: implement using sentence_log_likelihood."""
    raise NotImplementedError


# ---- Tier 2: API model, text-output only ----

def llm_judge_score(word, llm_response, judge_model_name):
    """
    LLM-as-judge scoring, modeled directly on the Language Models
    exercise (Problem 9): a second LLM call judges whether the
    response counts as knowing the word, rather than rigid
    exact/semantic string matching.
    TODO: implement after Aug 3 exercise (reuses llm_api.py).
    """
    raise NotImplementedError


def score_minimal_pair(llm_response, correct_choice):
    """
    Score whether the model picked the expected option.
    Note: randomize presentation order (or test both orders) to
    guard against position bias in LLM preference judgments.
    """
    raise NotImplementedError


# ---- Uncertainty (applies to both tiers) ----

def bootstrap_ci(values, n_boot=1000, ci=95):
    """
    Bootstrap resampling to get a confidence interval / std around
    any statistic computed on `values`. Use this before any
    cross-model or cross-dataset comparison is treated as
    meaningful - this was Dancho's explicit note on the ML
    project's 86% grade.

    TODO: implement (np.random.choice resampling + np.percentile).
    """
    raise NotImplementedError
