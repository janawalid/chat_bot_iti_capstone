import logging

import ollama

from app.core.config import settings

logger = logging.getLogger(__name__)


def build_prompt(question: str, retrieval_results: dict) -> str:
    documents = retrieval_results.get("documents", [[]])[0]
    metadatas = retrieval_results.get("metadatas", [[]])[0]

    context_blocks = []
    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "unknown")
        context_blocks.append(f"[Source: {source}]\n{doc}")
    context = "\n\n".join(context_blocks)

    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not contained in the context, say you don't know. "
        "Cite the source file(s) you used at the end of your answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def rewrite_query(question: str, history: list[dict], max_turns: int = 6) -> str:
    """Rewrite a possibly-ambiguous follow-up question (e.g. "explain how it
    works") into a standalone question that makes sense without the prior
    conversation, using the last `max_turns` messages as context.

    Falls back to returning the original question unchanged if there's no
    history yet, or if the rewrite call fails for any reason -- retrieval
    should never be blocked by this step.
    """
    if not history:
        return question

    recent = history[-max_turns:]
    history_text = "\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent)

    prompt = (
        "You are rewriting a user's follow-up question so it can be understood "
        "on its own, without needing the conversation history below. Use the "
        "conversation to resolve pronouns and implicit references (e.g. \"it\", "
        "\"that\", \"explain more\") into an explicit, standalone question.\n\n"
        "Rules:\n"
        "- If the follow-up question is already standalone, return it unchanged.\n"
        "- Do not answer the question.\n"
        "- Output ONLY the rewritten question, with no preamble, quotes, or explanation.\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )

    try:
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        rewritten = response["message"]["content"].strip().strip('"')
        if rewritten:
            logger.info("Rewrote query %r -> %r", question, rewritten)
            return rewritten
    except Exception:
        logger.exception("Query rewriting failed; falling back to original question")

    return question


def generate_answer(question: str, retrieval_results: dict) -> tuple[str, list[str]]:
    """Build the grounded prompt, call the local Ollama LLM, and return (answer, sources)."""
    prompt = build_prompt(question, retrieval_results)

    try:
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response["message"]["content"]
    except Exception:
        logger.exception("Ollama generation failed")
        raise

    metadatas = retrieval_results.get("metadatas", [[]])[0]
    sources = sorted({m.get("source", "unknown") for m in metadatas})

    return answer, sources