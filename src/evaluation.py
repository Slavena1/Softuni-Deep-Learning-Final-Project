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
from sklearn.metrics import r2_score


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

def bootstrap_ci(values, statistic=np.mean, n_boot=1000, ci=95, random_state=42):
    """
    Bootstrap resampling to get a confidence interval around any
    statistic computed on `values` (default: the mean). General
    purpose - use for per-word errors, feature importances, or
    anything else that isn't paired model predictions (for that,
    use bootstrap_r2_diff below instead).

    Returns (point_estimate, lower, upper).
    """
    rng = np.random.RandomState(random_state)
    values = np.asarray(values)
    n = len(values)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.randint(0, n, size=n)]
        boot_stats[i] = statistic(sample)
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    return (statistic(values),
            np.percentile(boot_stats, lower_pct),
            np.percentile(boot_stats, upper_pct))


def bootstrap_r2_ci(y_true, y_pred, n_boot=1000, ci=95, random_state=42):
    """
    Bootstrap CI on R^2 itself, by resampling (y_true, y_pred)
    pairs together with replacement and recomputing R^2 each time.
    Gives an honest uncertainty range around a single model's
    reported test R^2 - small test sets can produce R^2 estimates
    that swing a lot between resamples.

    Returns (point_estimate, lower, upper).
    """
    rng = np.random.RandomState(random_state)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    boot_r2 = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_r2[i] = r2_score(y_true[idx], y_pred[idx])
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    return (r2_score(y_true, y_pred),
            np.percentile(boot_r2, lower_pct),
            np.percentile(boot_r2, upper_pct))


def bootstrap_r2_diff(y_true, y_pred_a, y_pred_b, n_boot=1000, ci=95, random_state=42):
    """
    Paired bootstrap on the DIFFERENCE in R^2 between two models'
    predictions on the SAME test set. Resamples the same indices
    for both models each time (paired, not independent), since
    both were evaluated on the same words.

    Returns (point_estimate_of_diff, lower, upper). If the CI
    excludes zero, the difference is unlikely to be due to chance
    given this test set - this is the actual answer to "is this
    difference real or noise," not just eyeballing two R^2 numbers.
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)
    boot_diff = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_diff[i] = (r2_score(y_true[idx], y_pred_a[idx])
                         - r2_score(y_true[idx], y_pred_b[idx]))
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    point = r2_score(y_true, y_pred_a) - r2_score(y_true, y_pred_b)
    return point, np.percentile(boot_diff, lower_pct), np.percentile(boot_diff, upper_pct)
