import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.query import QueryRequest, QueryResponse
from app.services import retrieval, generation, guardrails

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "vector_store_ready": retrieval.is_ready()}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not retrieval.is_ready():
        raise HTTPException(status_code=503, detail="Vector store is not loaded yet")

    question = request.question.strip()

    # 1. Small talk — skip retrieval entirely, respond with a canned reply.
    #    Checked against the raw question; rewriting a smalltalk phrase like
    #    "thanks!" against history doesn't make sense.
    if guardrails.is_smalltalk(question):
        return QueryResponse(answer=guardrails.SMALLTALK_REPLY, sources=[])

    # 2. Query rewriting — resolve follow-up questions ("explain how it
    #    works") into a standalone question using recent conversation
    #    history, BEFORE retrieval. Without this, retrieval has no way to
    #    know what "it" refers to and searches on the literal follow-up
    #    text instead, which can pull in completely unrelated chunks.
    history = [turn.model_dump() for turn in request.history]
    standalone_question = generation.rewrite_query(question, history)

    # 3. Typo correction — using a dictionary built from the corpus itself.
    #    Runs on the *rewritten* question, since that's what actually gets
    #    embedded and searched.
    corrected_question = guardrails.correct_typos(standalone_question, retrieval.get_spellchecker())
    if corrected_question != standalone_question:
        logger.info("Corrected query %r -> %r", standalone_question, corrected_question)

    try:
        results = retrieval.retrieve(corrected_question)
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # 4. Out-of-scope — if nothing retrieved is actually close, say so
    # instead of forcing an answer out of irrelevant context.
    distances = results.get("distances", [[]])[0]
    if guardrails.is_out_of_scope(distances, settings.out_of_scope_threshold):
        return QueryResponse(answer=guardrails.OUT_OF_SCOPE_REPLY, sources=[])

    try:
        answer, sources = generation.generate_answer(corrected_question, results)
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return QueryResponse(answer=answer, sources=sources)