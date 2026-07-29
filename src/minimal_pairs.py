"""
minimal_pairs.py

ZhoBLiMP-style minimal pair generation, adapted from grammaticality
judgment to word-difficulty judgment - the same methodology already
used by the Chinese BabyLM pipeline's NLU track (grammar), applied
here to acquisition difficulty instead.

A pair here is: two sentences identical except one uses an
"easier" word and the other a "harder" one in the same slot.
"""


def generate_minimal_pair(easy_word, hard_word, template_sentence):
    """
    TODO: implement. Consider sourcing templates from real corpus
    sentences (BabyLM Chinese corpus) rather than hand-writing all
    of them, for ecological validity.
    """
    raise NotImplementedError


def build_minimal_pair_set(word_pairs, templates):
    """Batch-build a full test set of minimal pairs. TODO: implement."""
    raise NotImplementedError
