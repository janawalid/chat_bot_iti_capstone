# RAG Document Assistant — ML Concepts (Core Track)

A Retrieval-Augmented Generation (RAG) web application that answers questions about
machine learning concepts, grounded in a collection of Wikipedia articles, using a
local Ollama LLM. Built for the Level 2 Summer Training graduation project (Core Track).

## Overview

The user asks a question in a Streamlit chat UI → the FastAPI backend retrieves the most
relevant chunks from a persisted Chroma vector store → a prompt combining the retrieved
context and the question is sent to a local Ollama model → the grounded answer and its
source document(s) are returned and displayed.

## Architecture

```
┌─────────────┐      HTTP       ┌──────────────┐      retrieve      ┌───────────────┐
│  Streamlit  │ ───────────────▶│   FastAPI    │ ───────────────────▶│ Chroma vector │
│  frontend   │◀─────────────── │   backend    │◀─────────────────── │    store      │
└─────────────┘   answer+sources└──────┬───────┘                    └───────────────┘
                                        │  prompt(context, question)
                                        ▼
                                 ┌──────────────┐
                                 │  Ollama LLM  │
                                 │ (local model)│
                                 └──────────────┘
```

## Tech Stack

- **Data prep:** Python, `wikipedia` package, Jupyter Notebook
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store:** ChromaDB (persisted to disk)
- **LLM:** Ollama (local model, e.g. `llama3.2`)
- **Backend:** FastAPI, Pydantic, pytest
- **Frontend:** Streamlit

## Project Structure

```
rag-assistant-project/
├── collect_data.py            # Pulls Wikipedia articles into data/raw/
├── data/
│   ├── raw/                   # Source .txt documents (gitignored)
│   └── vector_store/          # Persisted Chroma DB (gitignored if large)
├── notebooks/
│   └── rag_pipeline.ipynb     # Chunking, embedding, retrieval, evaluation
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes/query.py
│   │   ├── core/config.py
│   │   ├── schemas/query.py
│   │   ├── services/retrieval.py
│   │   ├── services/generation.py
│   │   └── utils/logging_config.py
│   ├── data/vector_store/     # Copied from notebook output
│   ├── tests/test_query.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
└── frontend/
    ├── app.py
    ├── api_client.py
    ├── .env.example
    └── requirements.txt
```

## Domain & Data

The document collection covers ~20 core machine learning concepts (e.g. supervised
learning, gradient descent, neural networks, overfitting), sourced directly from
Wikipedia via `collect_data.py`. This produces clean, text-extractable `.txt` files with
no OCR required. To change domains, edit the `TOPICS` list in `collect_data.py` and
re-run it.

To (re)build the raw corpus:

```bash
pip install wikipedia
python collect_data.py
```

## Setup

### 1. Environment & data prep

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ollama pull llama3.2             # or another local model
python collect_data.py           # collects data/raw/*.txt

jupyter notebook notebooks/rag_pipeline.ipynb
# Run all cells top to bottom — this builds data/vector_store/
```

### 2. Backend

```bash
cd backend
cp .env.example .env             # adjust values if needed
pip install -r requirements.txt

# Copy the persisted vector store built by the notebook:
cp -r ../data/vector_store ./data/vector_store

uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

Run tests:

```bash
pytest
```

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

| Variable            | Description                                   | Example                  |
|---------------------|------------------------------------------------|---------------------------|
| `VECTOR_STORE_PATH` | Path to the persisted Chroma store              | `data/vector_store`       |
| `COLLECTION_NAME`   | Chroma collection name                          | `ml_docs`                 |
| `EMBEDDING_MODEL`   | Sentence-transformers model (must match notebook)| `all-MiniLM-L6-v2`        |
| `OLLAMA_MODEL`      | Local Ollama model used for generation          | `llama3.2`                |
| `RETRIEVAL_K`       | Number of chunks retrieved per query            | `3`                        |
| `CORS_ORIGINS`      | Allowed frontend origin(s), comma-separated     | `http://localhost:8501`   |

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

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is overfitting?"}'
```

```json
{
  "answer": "Overfitting occurs when a model learns noise in the training data rather than the underlying pattern... [Source: Overfitting.txt]",
  "sources": ["Overfitting.txt"]
}
```

## Evaluation Results

See `notebooks/rag_pipeline.ipynb`, section 2.6, for the full results table across 10 test
questions (question / retrieved source / answer / correct). Summary of failure modes and
mitigations is documented in the same section.

## Screenshots

_Add screenshots of the running Streamlit app and Swagger UI here before submission._

## Notes on Reproducing from Scratch

This README was verified by cloning the repository into a fresh folder and following only
the steps above.
