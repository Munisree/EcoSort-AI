"""
rag/retriever.py

Lightweight knowledge-base retrieval for EcoSort AI — Phase 2.

Loads waste_guidance.json once at module import and provides two retrieval
functions:

  get_guidance_by_category(category)
      Exact-match lookup by classifier output label.
      Returns the matching knowledge-base entry as a dict, or None.

  get_guidance_by_query(query)
      Keyword-based lookup for natural-language queries.
      Scores each knowledge-base entry by counting how many of the query's
      words appear in that entry's text fields, then returns the best match.
      Returns (entry_dict, score) — score is 0 if no match is found.

No external APIs, LLMs, or vector databases are used.
This module is intentionally simple and appropriate for a CPU-only prototype.

Phase 3 upgrade path:
  Replace get_guidance_by_query with a sentence-transformers embedding search
  against a FAISS index for semantic retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Knowledge-base path
# ---------------------------------------------------------------------------

_KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "waste_guidance.json"

# ---------------------------------------------------------------------------
# Load knowledge base once at import time
# ---------------------------------------------------------------------------

def _load_knowledge_base(path: Path) -> List[Dict]:
    """
    Load and parse waste_guidance.json.

    Returns a list of entry dicts, each with keys:
        category, short_description, disposal_guidance,
        safety_note, recycling_or_reuse_tip

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If the JSON is malformed or missing required keys.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {path}\n"
            "Expected file: knowledge_base/waste_guidance.json"
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            f"waste_guidance.json must be a non-empty JSON array. Got: {type(data)}"
        )

    required_keys = {
        "category",
        "short_description",
        "disposal_guidance",
        "safety_note",
        "recycling_or_reuse_tip",
    }
    for i, entry in enumerate(data):
        missing = required_keys - entry.keys()
        if missing:
            raise ValueError(
                f"Entry {i} in waste_guidance.json is missing keys: {missing}"
            )

    return data


# Load once. Module-level variable is reused by all calls.
_KB: List[Dict] = _load_knowledge_base(_KB_PATH)

# Build a fast category→entry index (lower-cased keys for robustness).
_KB_INDEX: Dict[str, Dict] = {
    entry["category"].lower(): entry for entry in _KB
}

# Sorted list of all known categories, for display and validation.
KNOWN_CATEGORIES: List[str] = sorted(_KB_INDEX.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_guidance_by_category(category: str) -> Optional[Dict]:
    """
    Retrieve the knowledge-base entry for a given waste category.

    The lookup is case-insensitive and strips leading/trailing whitespace,
    so classifier output labels such as "brown-glass", "Brown-Glass", or
    "  plastic  " are all handled correctly.

    Parameters
    ----------
    category : str
        Waste category label as returned by the classifier
        (e.g. "battery", "brown-glass", "plastic").

    Returns
    -------
    dict or None
        The matching knowledge-base entry with keys:
            category, short_description, disposal_guidance,
            safety_note, recycling_or_reuse_tip
        Returns None if the category is not found in the knowledge base.

    Examples
    --------
    >>> entry = get_guidance_by_category("plastic")
    >>> entry["disposal_guidance"]
    'Rinse plastic containers ...'
    """
    return _KB_INDEX.get(category.strip().lower())


def get_guidance_by_query(query: str) -> Tuple[Optional[Dict], int]:
    """
    Retrieve the most relevant knowledge-base entry for a natural-language
    query using lightweight keyword matching.

    Algorithm:
    1. Normalise the query: lower-case, remove punctuation, split into words.
    2. Remove common stop-words that carry no domain signal.
    3. For each knowledge-base entry, count how many query words appear in
       the concatenated text of all its fields.
    4. Return the entry with the highest count, together with that count.

    This is intentionally simple — no embeddings, no TF-IDF, no LLM.
    It works well for queries that contain waste-category names or obvious
    domain keywords (e.g. "how do I dispose of a battery?").

    Parameters
    ----------
    query : str
        A natural-language question or phrase from the user.

    Returns
    -------
    entry : dict or None
        The best-matching knowledge-base entry, or None if no query words
        matched anything (score == 0 and KB is empty).
    score : int
        Number of query words that matched the winning entry's text.
        A score of 0 means no meaningful match was found; the caller should
        treat the result as a low-confidence fallback.

    Examples
    --------
    >>> entry, score = get_guidance_by_query("what do I do with an old battery?")
    >>> entry["category"]
    'battery'
    >>> score > 0
    True
    """
    # ------------------------------------------------ Normalise query
    normalised = query.lower()
    # Remove punctuation except hyphens (category names use hyphens).
    normalised = re.sub(r"[^\w\s-]", " ", normalised)
    words = normalised.split()

    # Remove common English stop-words that do not carry domain meaning.
    _STOP_WORDS = {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "with", "do", "i", "my", "this", "that",
        "what", "how", "can", "should", "be", "are", "was", "were",
        "have", "has", "had", "will", "would", "could", "please",
        "tell", "me", "about", "some", "any", "which", "where", "old",
        "used", "use", "item", "thing", "stuff", "need", "want", "help",
    }
    query_words = [w for w in words if w not in _STOP_WORDS and len(w) > 1]

    if not query_words:
        # All words were stop-words; fall back to returning no match.
        return None, 0

    # ------------------------------------------------ Score each entry
    best_entry: Optional[Dict] = None
    best_score: int = 0

    for entry in _KB:
        # Build a single searchable text blob from all text fields.
        blob = " ".join([
            entry.get("category", ""),
            entry.get("short_description", ""),
            entry.get("disposal_guidance", ""),
            entry.get("safety_note", ""),
            entry.get("recycling_or_reuse_tip", ""),
        ]).lower()

        score = sum(1 for word in query_words if word in blob)

        if score > best_score:
            best_score = score
            best_entry = entry

    return best_entry, best_score


def list_categories() -> List[str]:
    """
    Return a sorted list of all category names in the knowledge base.

    Useful for displaying available categories in the UI or for validation.

    Returns
    -------
    list[str]
        Alphabetically sorted category names.
    """
    return KNOWN_CATEGORIES.copy()
