from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.retrieval import load_vector_store
from app.utils.logging_config import configure_logging



@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    # Load the vector store and embedding model once at startup, not per-request.
    load_vector_store()
    yield


app = FastAPI(title="RAG Document Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
