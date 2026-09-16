import logging
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.services.guardrails import build_domain_spellchecker

logger = logging.getLogger(__name__)

_embedding_model: Optional[SentenceTransformer] = None
_collection = None
_spellchecker = None


def load_vector_store() -> None:
    """Load the embedding model and persisted Chroma collection once at startup,
    and build a domain-aware spellchecker from the corpus for typo correction."""
    global _embedding_model, _collection, _spellchecker

    logger.info("Loading embedding model: %s", settings.embedding_model)
    _embedding_model = SentenceTransformer(settings.embedding_model)

    logger.info("Loading vector store from: %s", settings.vector_store_path)
    client = chromadb.PersistentClient(path=settings.vector_store_path)
    _collection = client.get_or_create_collection(settings.collection_name)

    count = _collection.count()
    if count == 0:
        logger.warning(
            "Vector store loaded but contains 0 chunks. "
            "Did you copy data/vector_store from the notebook output?"
        )
    else:
        logger.info("Vector store ready with %d chunks", count)

    # Build the typo-correction dictionary from the corpus itself, so domain
    # jargon (e.g. "autoencoder", "hyperparameter") isn't miscorrected.
    try:
        all_docs = _collection.get(include=["documents"])
        corpus_texts = all_docs.get("documents", []) or []
        _spellchecker = build_domain_spellchecker(corpus_texts)
        logger.info(
            "Domain spellchecker built from %d chunks (%d unique words)",
            len(corpus_texts),
            len(_spellchecker.word_frequency.dictionary),
        )
    except Exception:
        logger.exception("Failed to build domain spellchecker; typo correction will be skipped")
        _spellchecker = None


def is_ready() -> bool:
    return _embedding_model is not None and _collection is not None


def get_spellchecker():
    return _spellchecker


def retrieve(question: str, k: Optional[int] = None) -> dict:
    """Retrieve the top-k most relevant chunks for a question, including
    similarity distances so callers can detect out-of-scope questions."""
    if not is_ready():
        raise RuntimeError("Vector store not loaded. Call load_vector_store() first.")

    k = k or settings.retrieval_k
    query_embedding = _embedding_model.encode([question]).tolist()
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return results
