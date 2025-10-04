from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from modules.embedder import EmbeddingClient
from core.config import get_settings
import uvicorn

class EmbeddingResponse(BaseModel):
    embedding: list[float]

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.host_address = settings.host_address
    app.state.doc_embedder_client = EmbeddingClient(settings.api_document_embedding, settings.model_name)
    app.state.query_embedder_client = EmbeddingClient(settings.api_query_embedding, settings.model_name)
    yield
    app.state.doc_embedder_client.close()
    app.state.query_embedder_client.close()
    

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# usage host:8000/embed_document?text=your_text_here
@app.get("/embed_document", response_model=EmbeddingResponse)
async def embed_document(text: str):
    res = await app.state.doc_embedder_client.embed_document(text)
    return {"embedding": res}
@app.get("/embed_query", response_model=EmbeddingResponse)
async def embed_query(text: str):
    res = await app.state.query_embedder_client.embed_query(text)
    return {"embedding": res}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)