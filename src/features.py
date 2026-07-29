"""
features.py
"""


def spectrum_score(r_freq, r_conc):
    """
    Compute a spectrum score from 0 (concreteness-driven)
    to 1 (frequency-driven), using absolute Spearman correlations.
    Absolute values are used because both correlations are negative
    (higher frequency and higher concreteness both predict earlier
    acquisition), so the sign reflects direction, not dominance.
    The language model scores 1.0 by definition.
    """
    denom = abs(r_freq) + abs(r_conc)
    if denom == 0:
        return 0.0
    return abs(r_freq) / denom


