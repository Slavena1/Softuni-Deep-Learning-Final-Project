"""
llm_api.py

LLM API integration. Provides a unified interface for quering LLMs
about Mandarin vocabulary, with caching, retry logic, and cost
tracking. Supports Claude, OpenAI and Qwen through provider-specific callers.

Structured response returned by all providers.
"""

import json
import os
import time
import random
from pydantic import BaseModel

CACHE_DIR = "../data/llm_responses"
COST_LOG_PATH = "../data/cost_log.csv"

# Rough per-model pricing, USD per 1M tokens (input, output) - approximate,
# for cost tracking purposes only, not billing-accurate. Update to match
# whatever specific models you actually use.
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "qwen-plus": (0.4, 1.2),  # via OpenRouter - verify actual rate there
}

WORD_KNOWLEDGE_PROMPT = """You are being asked about a single Mandarin Chinese word, as part of a linguistics research study on vocabulary acquisition.

Word: {word}

Respond with your assessment of this word:
- confidence: how confident are you that you know this word well, from 0.0 (never seen it) to 1.0 (know it perfectly)
- definition: a brief definition (English), or empty string if you don't know the word
- example_sentence: a natural Mandarin sentence using this word, or empty string if you can't
- can_use_in_sentence: true if you could confidently use this word correctly in a sentence, false otherwise
"""


class WordKnowledge(BaseModel):
    """Pydantic schema for OpenAI's Responses API structured output (client.responses.parse)."""
    confidence: float
    definition: str
    example_sentence: str
    can_use_in_sentence: bool


# ---- Core plumbing (tested) ----

def _cache_path(model_name, word):
    safe_model = model_name.replace("/", "_")
    return os.path.join(CACHE_DIR, safe_model, f"{word}.json")


def sanitize_input(word):
    """Strip whitespace, enforce a sane length ceiling - guards against
    malformed word-list entries reaching the API."""
    word = str(word).strip()
    if not word or len(word) > 20:
        raise ValueError(f"Word failed sanitization (empty or too long): {word!r}")
    return word


def log_call(model_name, word, tokens_in, tokens_out, cost, cached, timestamp=None):
    """Append one call record to data/cost_log.csv, creating it with a header if needed."""
    os.makedirs(os.path.dirname(COST_LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(COST_LOG_PATH)
    ts = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
    with open(COST_LOG_PATH, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("timestamp,model,word,tokens_in,tokens_out,cost_usd,cached\n")
        f.write(f"{ts},{model_name},{word},{tokens_in},{tokens_out},{cost:.6f},{cached}\n")


def _estimate_cost(model_name, tokens_in, tokens_out):
    price_in, price_out = MODEL_PRICING.get(model_name, (0.0, 0.0))
    return (tokens_in / 1_000_000) * price_in + (tokens_out / 1_000_000) * price_out


def query_word_knowledge(word, provider, caller, model_name, use_cache=True,
                          max_retries=3, base_delay=2):
    """
    Core function: ask `caller` (a provider-specific function taking
    (word, model_name) and returning (result_dict, tokens_in, tokens_out))
    what it knows about `word`. Cached to disk on first fetch, retried
    with exponential backoff on transient failures.
    """
    word = sanitize_input(word)
    cache_file = _cache_path(f"{provider}_{model_name}", word)

    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_result = json.load(f)
        log_call(model_name, word, 0, 0, 0.0, cached=True)
        return cached_result

    last_error = None
    for attempt in range(max_retries):
        try:
            result, tokens_in, tokens_out = caller(word, model_name)
            cost = _estimate_cost(model_name, tokens_in, tokens_out)
            log_call(model_name, word, tokens_in, tokens_out, cost, cached=False)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
    raise RuntimeError(f"Failed to query {word!r} via {provider}/{model_name} "
                        f"after {max_retries} attempts: {last_error}")


def batch_query(word_list, provider, caller, model_name, use_cache=True,
                 rate_limit_delay=0.5, verbose=True):
    """Run query_word_knowledge over a full word list, with a simple
    fixed delay between non-cached calls, and a progress indicator."""
    results = {}
    for i, word in enumerate(word_list):
        if verbose and i % 25 == 0:
            print(f"  {i}/{len(word_list)} ({provider}/{model_name})...")
        was_cached = use_cache and os.path.exists(_cache_path(f"{provider}_{model_name}", word))
        results[word] = query_word_knowledge(word, provider, caller, model_name, use_cache=use_cache)
        if not was_cached:
            time.sleep(rate_limit_delay)
    if verbose:
        print(f"  {len(word_list)}/{len(word_list)} done.")
    return results


# ---- Provider-specific callers (written carefully, unverified live - see note above) ----

def make_claude_caller(client):
    """
    Claude structured output via forced tool use - the standard reliable
    pattern for valid JSON from Claude, since there's no separate
    JSON-mode flag the way OpenAI has.
    """
    tool = {
        "name": "word_knowledge",
        "description": "Report your knowledge of a Mandarin word.",
        "input_schema": {
            "type": "object",
            "properties": {
                "confidence": {"type": "number"},
                "definition": {"type": "string"},
                "example_sentence": {"type": "string"},
                "can_use_in_sentence": {"type": "boolean"},
            },
            "required": ["confidence", "definition", "example_sentence", "can_use_in_sentence"],
        },
    }

    def caller(word, model_name):
        response = client.messages.create(
            model=model_name,
            max_tokens=300,
            tools=[tool],
            tool_choice={"type": "tool", "name": "word_knowledge"},
            messages=[{"role": "user", "content": WORD_KNOWLEDGE_PROMPT.format(word=word)}],
        )
        tool_use_block = next(b for b in response.content if b.type == "tool_use")
        return tool_use_block.input, response.usage.input_tokens, response.usage.output_tokens

    return caller


def make_openai_direct_caller(client):
    """OpenAI direct, via the Responses API + Pydantic schema (the
    exact pattern shown in this week's lecture, slide 17)."""

    def caller(word, model_name):
        response = client.responses.parse(
            model=model_name,
            input=WORD_KNOWLEDGE_PROMPT.format(word=word),
            text_format=WordKnowledge,
        )
        result = response.output_parsed.model_dump()
        return result, response.usage.input_tokens, response.usage.output_tokens

    return caller


def make_qwen_openrouter_caller(client):
    """
    Qwen via OpenRouter. Uses the older Chat Completions shape
    (not the newer Responses API) deliberately - Chat Completions is
    the far more universally-supported compatibility target across
    OpenAI-compatible proxies; the newer Responses API's third-party
    support is much less certain.
    """
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "word_knowledge",
            "schema": {
                "type": "object",
                "properties": {
                    "confidence": {"type": "number"},
                    "definition": {"type": "string"},
                    "example_sentence": {"type": "string"},
                    "can_use_in_sentence": {"type": "boolean"},
                },
                "required": ["confidence", "definition", "example_sentence", "can_use_in_sentence"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    def caller(word, model_name):
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": WORD_KNOWLEDGE_PROMPT.format(word=word)}],
            response_format=schema,
        )
        result = json.loads(response.choices[0].message.content)
        return result, response.usage.prompt_tokens, response.usage.completion_tokens

    return caller
