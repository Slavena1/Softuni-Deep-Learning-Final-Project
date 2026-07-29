"""
llm_api.py

LLM API integration layer - covers the DL syllabus's "LLM API
Integration" section directly.

IMPLEMENTATION DEFERRED: design decisions below are locked, but the
actual code is being written after the Jul 30 lecture and Aug 3
exercise cover this material directly.

Design decisions already made:
- USE_CACHE = True by default in the notebook - every response
  cached to data/llm_responses/ on first fetch, so the notebook is
  fully reviewable with zero API cost.
- Two-tier methodology: Claude has no logprobs (verified), so all
  three providers (Claude, one OpenAI model, Qwen via DashScope)
  get behavioral-tier testing (sentence completion, minimal pairs);
  only OpenAI additionally supports the internal-access tier
  (log-likelihood via logprobs) - Qwen's DashScope endpoint does
  NOT support logprobs either (verified), despite being
  OpenAI-compatible.
- Scoring: LLM-as-judge, not exact/semantic string match - modeled
  on the Language Models exercise (Problem 9).
"""

import json
import os


CACHE_DIR = "../data/llm_responses"


def _cache_path(model_name, word):
    """Deterministic cache file path for one (model, word) pair."""
    safe_model = model_name.replace("/", "_")
    return os.path.join(CACHE_DIR, safe_model, f"{word}.json")


def query_word_difficulty(word, model_name, prompt_template, use_cache=True,
                           max_retries=3, timeout=30):
    """
    Core function: ask an LLM to complete a sentence / judge a
    minimal pair involving `word`, via structured JSON output.
    TODO: implement after Aug 3 exercise.
    """
    raise NotImplementedError


def sanitize_input(word):
    """
    Basic input sanitization before sending to the API.
    TODO: implement after Aug 3 exercise.
    """
    raise NotImplementedError


def log_call(model_name, word, tokens_in, tokens_out, cost, cached, timestamp=None):
    """
    Append one call record to data/cost_log.csv.
    TODO: implement after Aug 3 exercise.
    """
    raise NotImplementedError


def batch_query(word_list, model_name, prompt_template, **kwargs):
    """
    Run query_word_difficulty over a full word list, with a simple
    rate limiter and progress indicator.
    TODO: implement after Aug 3 exercise.
    """
    raise NotImplementedError
