from fastapi import FastAPI
from contextlib import asynccontextmanager

from .document_embedder import EmbeddingClient, get_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.doc_embedder_client = EmbeddingClient(settings.api_document_embedding, settings.model_name)
    app.state.query_embedder_client = EmbeddingClient(settings.api_query_embedding, settings.model_name)
    yield
    app.state.doc_embedder_client.close()
    app.state.query_embedder_client.close()
    

app = FastAPI(lifespan=lifespan)



@app.get("/health")
async def health_check():
    return {"status": "ok"}
