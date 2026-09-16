"""
Guardrails for the RAG pipeline:
  1. Typo correction — using a spelling dictionary built from the project's
     OWN document corpus, so domain jargon (e.g. "autoencoder") isn't
     incorrectly "corrected" into something else.
  2. Small talk detection — greetings/thanks/goodbyes skip retrieval
     entirely and get a canned, friendly reply.
  3. Out-of-scope detection — if the closest retrieved chunk is still too
     dissimilar (distance too high), the question is probably not covered
     by the document collection, so we say so instead of forcing an answer
     out of irrelevant context.
"""

import re
from typing import Optional

from spellchecker import SpellChecker

SMALLTALK_PATTERNS = [
    r"^(hi|hello|hey|yo|sup|hiya)[\s!.,]*$",
    r"^(how are you( doing)?|what'?s up|how'?s it going)[\s?!.,]*$",
    r"^(thanks|thank you|thx|ty|appreciate it)[\s!.,]*$",
    r"^(bye|goodbye|see ya|see you|later)[\s!.,]*$",
    r"^(ok|okay|cool|nice|great|awesome|lol|haha)[\s!.,]*$",
    r"^good (morning|afternoon|evening|night)[\s!.,]*$",
    r"^(who are you|what are you|what can you do)[\s?!.,]*$",
]

SMALLTALK_REPLY = (
    "Hi! I'm a document assistant focused on machine learning concepts. "
    "Ask me something like \"What is overfitting?\" or \"Explain gradient descent\", "
    "and I'll answer using the source documents I have indexed."
)

OUT_OF_SCOPE_REPLY = (
    "That looks like it's outside the topics covered in this document collection "
    "(machine learning concepts). Try asking about things like overfitting, "
    "gradient descent, neural networks, or cross-validation."
)


def is_smalltalk(query: str) -> bool:
    q = query.strip().lower()
    return any(re.match(p, q) for p in SMALLTALK_PATTERNS)


def build_domain_spellchecker(corpus_texts: list[str]) -> SpellChecker:
    """Build a spellchecker whose dictionary is seeded ONLY with words from
    the project's own documents (language=None disables pyspellchecker's
    built-in English dictionary). Without this, obscure dictionary words
    (e.g. "jurel", an actual fish name) can outrank real corrections like
    "neural" simply by being closer in edit-distance to a typo. distance=2
    allows fixing typos that are off by more than one character (e.g.
    "nurel" -> "neural", "ntwerk" -> "network").

    IMPORTANT: we load the raw corpus text with load_text() rather than a
    deduplicated set of unique words. When multiple candidate corrections
    are tied on edit-distance (e.g. "neural", "cure", "nurse", "curl" are
    all 2 edits from "nurel"), pyspellchecker breaks the tie using word
    FREQUENCY. A deduplicated set gives every word frequency 1, making the
    tie-break effectively random. Loading the real text preserves the fact
    that "neural" appears far more often in an ML corpus than "cure" does,
    so common domain terms correctly win ties over rare incidental words.
    """
    spellchecker = SpellChecker(language=None, distance=2)
    full_text = " ".join(corpus_texts).lower()
    spellchecker.word_frequency.load_text(full_text)
    return spellchecker


def correct_typos(query: str, spellchecker: Optional[SpellChecker]) -> str:
    """Best-effort correction of individual words not recognized by the
    domain dictionary. Leaves punctuation, casing structure, and unknown
    short tokens (e.g. acronyms) largely alone; only swaps words the
    spellchecker is confident about.
    """
    if spellchecker is None:
        return query

    corrected_words = []
    for word in query.split():
        clean = re.sub(r"[^a-zA-Z']", "", word.lower())
        # Skip very short tokens/acronyms (CNN, ML, k, etc.) — too risky to "correct"
        if len(clean) <= 2 or clean in spellchecker:
            corrected_words.append(word)
            continue
        suggestion = spellchecker.correction(clean)
        if suggestion and suggestion != clean:
            corrected_words.append(suggestion)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)


def is_out_of_scope(distances: list[float], threshold: float) -> bool:
    """Chroma returns L2 distances (lower = more similar) by default.
    If even the closest retrieved chunk is farther than `threshold`,
    treat the question as not covered by the corpus.
    """
    if not distances:
        return True
    return min(distances) > threshold