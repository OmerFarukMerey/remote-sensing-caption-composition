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


# Per-class lexical proxies for the caption-fidelity stratified test.
# Hand-built; intentionally lower-cased substring matches.
CLASS_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Tree":     ("tree", "wood", "forest"),
    "Shrub":    ("shrub", "bush"),
    "Grass":    ("grass", "pasture", "meadow", "prairie", "savanna"),
    "Crop":     ("crop", "cultivat", "farmland", "agricultur", "field", "plantation"),
    "Built-up": ("urban", "built", "building", "city", "road", "settlement", "village", "town", "house", "industrial"),
    "Barren":   ("barren", "rocky", "desert", "arid", "rock", "bare", "exposed", "bedrock", "soil", "sand", "dune"),
    "Water":    ("water", "river", "lake", "stream", "pond", "ocean", "sea", "channel", "riverbed", "wetland"),
}


def caption_describes_class(caption: str, class_name: str) -> bool:
    """Cheap lexical check: does any synonym for `class_name` appear in `caption`?"""
    if not caption or class_name not in CLASS_SYNONYMS:
        return False
    text = caption.lower()
    return any(syn in text for syn in CLASS_SYNONYMS[class_name])


def caption_describes_dominant_class(caption: str, gt_classes: list[str], gt_vec) -> bool:
    """Return True iff the caption mentions the GT dominant class (argmax of gt_vec).

    `gt_classes` is the ordered list of class names (length 7); `gt_vec` is a
    same-length array/list/tensor of percentages or fractions.
    """
    import numpy as np

    arr = np.asarray(gt_vec, dtype=float)
    dominant = gt_classes[int(arr.argmax())]
    return caption_describes_class(caption, dominant)
