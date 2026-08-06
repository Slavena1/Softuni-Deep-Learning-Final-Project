"""
evaluation.py

Two-tier "does the model know this word?" methodology:

  Tier 1 - local model (full internal access): log-likelihood,
  perplexity, masked-word prediction. The classic BabyLM-style
  AoA-prediction method.

  Tier 2 - API models, all three providers (no logprobs available
  for any of them - verified for both Claude and Qwen/DashScope):
  sentence completion and minimal-pair judgments, scored via
  LLM-as-judge, not exact/semantic string match.

Uncertainty Analysis (bootstrap)

"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, rankdata, pearsonr
import features as ft


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
    statistic computed on `values` (default: the mean).

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
    predictions on the SAME test set.

    Returns (point_estimate_of_diff, lower, upper).
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


def stratified_error_analysis(df, error_cols, strat_col, n_buckets=5, bucket_labels=None):
    """
    Bucket `df` by `strat_col` into `n_buckets` quantile groups, and
    compute a bootstrap CI on the mean of each column in error_cols
    within each bucket - the Section 7 stratification pattern
    (frequency, concreteness, word length...), factored into one
    reusable function instead of repeating it per variable.

    Returns a tidy DataFrame: one row per (bucket, model).
    """
    df = df.copy()
    df['_bucket'] = pd.qcut(df[strat_col], q=n_buckets, labels=bucket_labels, duplicates='drop')
    records = []
    for bucket, group in df.groupby('_bucket', observed=True):
        for col in error_cols:
            point, lo, hi = bootstrap_ci(group[col].values)
            records.append({'bucket': bucket, 'model': col, 'n': len(group),
                             'mean_error': round(point, 3),
                             'ci_lower': round(lo, 3), 'ci_upper': round(hi, 3)})
    return pd.DataFrame(records)


def score_word_knowledge(response):
    """
    Primary score: the model's own self-reported confidence (0-1).
    No separate judge call is needed for the core comparison, since
    the structured response IS the model's direct self-assessment -
    a cleaner signal than scoring free-text completions would have
    required.
    """
    return response['confidence']


def llm_judge_validate(word, response_to_check, judge_caller, judge_model_name):
    """
    Ask a separate judge model whether this model's self-reported definition/
    example_sentence for `word` is actually correct.
    Run on a random subset for validation, not every word.
    """
    prompt = (
        f"A language model was asked about the Mandarin word '{word}' and gave this "
        f"definition: '{response_to_check['definition']}' and example sentence: "
        f"'{response_to_check['example_sentence']}'. Is this correct? Answer only 'yes' or 'no'."
    )
    answer_text, _, _ = judge_caller(prompt, judge_model_name)
    return answer_text.strip().lower().startswith('y')


def compute_spectrum_score(df, value_col, freq_col='log_frequency', conc_col='concreteness',
                            n_boot=1000, ci=95, random_state=42):
    """
    Compute a spectrum score (0=concreteness-driven, 1=frequency-driven)
    for any value column against frequency and concreteness, via Spearman
    correlations - generalizes features.spectrum_score to work on any
    dataframe column, with a bootstrap CI.
    Returns a dict: r_freq, r_conc, spectrum_score, ci_lower, ci_upper.
    """
    import features as ft

    r_freq, _ = spearmanr(df[value_col], df[freq_col])
    r_conc, _ = spearmanr(df[value_col], df[conc_col])
    point_score = ft.spectrum_score(r_freq, r_conc)

    rng = np.random.RandomState(random_state)
    n = len(df)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sample = df.iloc[idx]
        if sample[value_col].nunique() < 2:
            continue
        rf, _ = spearmanr(sample[value_col], sample[freq_col])
        rc, _ = spearmanr(sample[value_col], sample[conc_col])
        boot_scores.append(ft.spectrum_score(rf, rc))

    lower_pct, upper_pct = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        'r_freq': r_freq, 'r_conc': r_conc, 'spectrum_score': point_score,
        'ci_lower': np.percentile(boot_scores, lower_pct),
        'ci_upper': np.percentile(boot_scores, upper_pct),
    }


def compute_partial_spectrum_score(df, value_col, freq_col='log_frequency', conc_col='concreteness',
                                    n_boot=1000, ci=95, random_state=42):
    """
    Partial Spearman correlation version of compute_spectrum_score:
    isolates each feature's independent contribution, controlling for
    the other - closer in spirit to what Lasso measures, and a
    robustness check against collinearity inflating the raw version.
    """
    def partial_corrs(sub_df):
        v = rankdata(sub_df[value_col]); f = rankdata(sub_df[freq_col]); c = rankdata(sub_df[conc_col])
        r_vf, _ = pearsonr(v, f); r_vc, _ = pearsonr(v, c); r_fc, _ = pearsonr(f, c)
        denom_f = np.sqrt((1 - r_vc**2) * (1 - r_fc**2))
        denom_c = np.sqrt((1 - r_vf**2) * (1 - r_fc**2))
        pr_vf = (r_vf - r_vc*r_fc) / denom_f if denom_f > 0 else 0.0
        pr_vc = (r_vc - r_vf*r_fc) / denom_c if denom_c > 0 else 0.0
        return pr_vf, pr_vc

    pr_vf, pr_vc = partial_corrs(df)
    point_score = ft.spectrum_score(pr_vf, pr_vc)

    rng = np.random.RandomState(random_state)
    n = len(df)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sample = df.iloc[idx]
        if sample[value_col].nunique() < 2:
            continue
        try:
            bpr_vf, bpr_vc = partial_corrs(sample)
            boot_scores.append(ft.spectrum_score(bpr_vf, bpr_vc))
        except Exception:
            continue
    lower_pct, upper_pct = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {'partial_r_freq': pr_vf, 'partial_r_conc': pr_vc, 'partial_spectrum_score': point_score,
            'ci_lower': np.percentile(boot_scores, lower_pct),
            'ci_upper': np.percentile(boot_scores, upper_pct)}
