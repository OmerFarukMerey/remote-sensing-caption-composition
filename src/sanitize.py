"""Text sanitization and keyword extraction for the Information Richness ablation.

Phase 2 uses only `vision_*` captions (no GT exposure). Numbers are still stripped
as a safety filter before tokenization to honor the no-leakage commitment.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

NUM_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?")
WS_PATTERN = re.compile(r"\s{2,}")


def mask_numbers(text: str, replacement: str = "[NUM]") -> str:
    """Replace integers, decimals, and percentages with a placeholder.

    Used as a safety filter on vision_* captions, which should be number-free
    in principle but may contain hallucinated digits.
    """
    if text is None:
        return ""
    out = NUM_PATTERN.sub(replacement, str(text))
    out = WS_PATTERN.sub(" ", out).strip()
    return out


@lru_cache(maxsize=1)
def _load_spacy():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc


def extract_keywords(captions: Iterable[str]) -> str:
    """Return a comma-separated string of unique noun lemmas from captions.

    Pre-masks numbers before parsing so noun_chunks aren't polluted by
    accidental digits. Empty/None captions are skipped silently.
    """
    nlp = _load_spacy()
    chunks: set[str] = set()
    for cap in captions:
        if not cap:
            continue
        doc = nlp(mask_numbers(cap))
        for nc in doc.noun_chunks:
            head = nc.root.lemma_.lower()
            if head.isalpha() and len(head) > 2:
                chunks.add(head)
    return ", ".join(sorted(chunks))
