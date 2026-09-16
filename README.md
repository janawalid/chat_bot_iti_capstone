# RAG Document Assistant — ML Concepts (Core Track)

A Retrieval-Augmented Generation (RAG) web application that answers questions about
machine learning concepts, grounded in a collection of Wikipedia articles, using a
local Ollama LLM. Built for the Level 2 Summer Training graduation project (Core Track).

## Overview

The user asks a question in a Streamlit chat UI → the FastAPI backend rewrites
follow-up questions using conversation history, runs typo correction and small-talk /
out-of-scope checks, then retrieves the most relevant chunks from a persisted Chroma
vector store → a prompt combining the retrieved context and the question is sent to a
local Ollama model → the grounded answer and its source document(s) are returned and
displayed.

## Architecture

```
┌─────────────┐      HTTP       ┌──────────────┐      retrieve      ┌───────────────┐
│  Streamlit  │ ───────────────▶│   FastAPI    │ ───────────────────▶│ Chroma vector │
│  frontend   │◀─────────────── │   backend    │◀─────────────────── │    store      │
└─────────────┘   answer+sources└──────┬───────┘                    └───────────────┘
                                        │  rewrite → typo-fix → retrieve →
                                        │  prompt(context, question)
                                        ▼
                                 ┌──────────────┐
                                 │  Ollama LLM  │
                                 │ (local model)│
                                 └──────────────┘
```

## Tech Stack

- **Data prep:** Python, Wikipedia REST API via `requests`, Jupyter Notebook
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store:** ChromaDB (persisted to disk)
- **LLM:** Ollama (local model, e.g. `llama3.2`)
- **Guardrails:** `pyspellchecker` (domain-dictionary typo correction), regex small-talk
  detection, distance-based out-of-scope detection
- **Backend:** FastAPI, Pydantic, pytest
- **Frontend:** Streamlit

## Project Structure

```
chat_bot_iti_capstone/
├── collect_data.py            # Pulls Wikipedia articles into data/raw/ via the Wikipedia API
├── data/
│   └── raw/                   # Source .txt documents (gitignored)
├── notebooks/
│   └── rag_pipeline.ipynb     # Chunking, embedding, retrieval, guardrails, evaluation
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes/query.py
│   │   ├── core/config.py
│   │   ├── schemas/query.py
│   │   ├── services/retrieval.py
│   │   ├── services/generation.py
│   │   └── services/guardrails.py
│   ├── data/vector_store/     # Persisted Chroma DB, committed (~13 MB)
│   ├── tests/test_query.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── .env.example
│   └── requirements.txt
└── requirements.txt            # Notebook/data-prep dependencies only
```

> **Note:** `backend/data/vector_store/` is the one the API actually loads and is
> committed to the repo. There is no second copy at the project root — the notebook
> writes directly to `backend/data/vector_store/` (see Setup below).

## Domain & Data

The document collection covers 20 core machine learning concepts (e.g. supervised
learning, gradient descent, neural networks, overfitting), sourced from Wikipedia via
`collect_data.py`. This calls the Wikipedia API directly with `requests` and a
descriptive User-Agent — the `wikipedia` PyPI package is intentionally **not** used,
since it doesn't send a proper User-Agent and Wikipedia now blocks such requests. This
produces clean, text-extractable `.txt` files with no OCR required. To change domains,
edit the `TOPICS` list in `collect_data.py` and re-run it.

To (re)build the raw corpus:

```bash
pip install requests
python collect_data.py
```

## Guardrails

Beyond plain retrieve-then-generate, `/query` runs three checks before calling the LLM
(implemented in `backend/app/services/guardrails.py`, developed in notebook section 2.4b):

1. **Query rewriting** — a follow-up question like "explain how it works" is rewritten
   into a standalone question using recent conversation `history`, so retrieval isn't
   run on an ambiguous pronoun.
2. **Typo correction** — corrects individual words using a spelling dictionary built
   *from the project's own corpus* rather than a general English dictionary, so domain
   jargon (e.g. "autoencoder", "hyperparameter") isn't miscorrected into an unrelated
   word.
3. **Small-talk and out-of-scope handling** — greetings/thanks get a canned reply
   without hitting the vector store; questions whose closest retrieved chunk is still
   too dissimilar (L2 distance above `OUT_OF_SCOPE_THRESHOLD`) get a "this isn't in my
   document collection" reply instead of a forced, likely-hallucinated answer.

## Setup

### 1. Environment & data prep

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ollama pull llama3.2             # or another local model
python collect_data.py           # collects data/raw/*.txt

jupyter notebook notebooks/rag_pipeline.ipynb
# Run all cells top to bottom — this builds backend/data/vector_store/
```

The persisted vector store is already committed to `backend/data/vector_store/`, so you
can skip straight to step 2 to run the app as-is. Re-run the notebook only if you change
the domain, the chunking strategy, or the embedding model.

### 2. Backend

```bash
cd backend
cp .env.example .env             # adjust values if needed
pip install -r requirements.txt

uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

Run tests:

```bash
pytest
```

Or run the backend in Docker instead:

```bash
cd backend
docker build -t rag-backend .
docker run -p 8000:8000 --env-file .env rag-backend
```

> Ollama runs on the host, not in the container. On Linux, add
> `--add-host=host.docker.internal:host-gateway` and point `OLLAMA_HOST` at
> `http://host.docker.internal:11434` if your Ollama client doesn't already default to it.

### 3. Frontend

```bash
cd frontend
cp .env.example .env             # set API_BASE_URL if different
pip install -r requirements.txt

streamlit run app.py
# Opens at http://localhost:8501
```

## Environment Variables

### Backend (`backend/.env`)

| Variable                 | Description                                                          | Example                 |
|---------------------------|------------------------------------------------------------------------|---------------------------|
| `VECTOR_STORE_PATH`       | Path to the persisted Chroma store                                     | `data/vector_store`       |
| `COLLECTION_NAME`         | Chroma collection name                                                  | `ml_docs`                 |
| `EMBEDDING_MODEL`         | Sentence-transformers model (must match notebook)                     | `all-MiniLM-L6-v2`        |
| `OLLAMA_MODEL`            | Local Ollama model used for generation                                 | `llama3.2`                |
| `RETRIEVAL_K`             | Number of chunks retrieved per query                                    | `3`                        |
| `CORS_ORIGINS`            | Allowed frontend origin(s), comma-separated                            | `http://localhost:8501`   |
| `OUT_OF_SCOPE_THRESHOLD`  | Max L2 distance for the closest chunk before a question is treated as out-of-scope | `1.3`                      |

### Frontend (`frontend/.env`)

| Variable        | Description                     | Example                 |
|-----------------|----------------------------------|--------------------------|
| `API_BASE_URL`  | URL of the running backend API   | `http://localhost:8000`  |

## API Reference

### `GET /health`

Returns service status.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "vector_store_ready": true}
```

### `POST /query`

Retrieves relevant context, generates a grounded answer, and returns it with sources.
`history` is optional — pass recent turns so the backend can resolve follow-up
questions (e.g. "explain more") before retrieval.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is overfitting?",
    "history": []
  }'
```

```json
{
  "answer": "Overfitting occurs when a model learns noise in the training data rather than the underlying pattern... [Source: Overfitting.txt]",
  "sources": ["Overfitting.txt"]
}
```

## Evaluation Results

Ran against 10 test questions in notebook section 2.6 (`notebooks/rag_pipeline.ipynb`),
with `k=3`. Each answer was manually checked against the source article for grounding.

| # | Question | Retrieved source(s) | Correct? |
|---|----------|----------------------|----------|
| 1 | What is overfitting and how can it be reduced? | Overfitting.txt, Overfitting.txt, Convolutional_neural_network.txt | True |
| 2 | How does gradient descent minimize a loss function? | Gradient_descent.txt, Gradient_descent.txt, Supervised_learning.txt | True |
| 3 | What is the difference between supervised and unsupervised learning? | Unsupervised_learning.txt ×3 | True |
| 4 | What is a convolutional neural network used for? | Convolutional_neural_network.txt ×3 | True |
| 5 | Explain the bias-variance tradeoff. | Supervised_learning.txt, Bias-variance_tradeoff.txt ×2 | True |
| 6 | What is transfer learning? | Transfer_learning.txt ×3 | True |
| 7 | How does a random forest differ from a single decision tree? | Random_forest.txt, Decision_tree_learning.txt, Random_forest.txt | **False** |
| 8 | What is an autoencoder used for? | Autoencoder.txt ×3 | True |
| 9 | What is reinforcement learning? | Machine_learning.txt, Reinforcement_learning.txt ×2 | True |
| 10 | What is cross-validation and why is it used? | Cross-validation_statistics.txt ×3 | True |

**Result: 9/10 correct (90%).**

**Failure cases observed:** Question 7 ("random forest vs decision tree") is a direct
example of the multi-document coverage problem — at k=3 the retriever pulled 2 chunks
from Random_forest.txt and only 1 from Decision_tree_learning.txt, and the model
responded "I don't know" rather than synthesize a partial comparison. Increasing k to
4–5 for comparison-style questions, or retrieving per-entity before merging, would
likely fix this. A second, milder failure mode was minor terseness on narrow factual
questions (e.g. Q6) — the answers were correct but minimal, which a slightly more
detailed prompt instruction could improve.

## Screenshots
<img width="1535" height="730" alt="image" src="https://github.com/user-attachments/assets/f9b5a017-2679-4b25-9d26-fbc95e52349f" />

### out of scope
<img width="1535" height="717" alt="image" src="https://github.com/user-attachments/assets/e95eee1c-79f7-420f-9d86-282c944ca8b9" />

### side talk and typos
<img width="1534" height="702" alt="image" src="https://github.com/user-attachments/assets/bdce9ca0-8f6a-445b-ab50-826f55e9bdcf" />
