import requests
from langchain_community.embeddings import JinaEmbeddings
from dotenv import load_dotenv
import os
from pydantic import SecretStr
import requests
MODEL_NAME = "jina-embeddings-v4"
load_dotenv()
_session = requests.Session()


# Safety first
_doc_embedder = os.getenv("API_DOCUMENT_EMBEDDING")
if not _doc_embedder:
    raise ValueError("API_DOCUMENT_EMBEDDING environment variable is not set")
DOC_EMBEDDER: SecretStr = SecretStr(_doc_embedder)

_query_embedder = os.getenv("QUERY_EMBEDDING")
if not _query_embedder:
    raise ValueError("QUERY_EMBEDDING environment variable is not set")
QUERY_EMBEDDER: SecretStr = SecretStr(_query_embedder)


async def embed_document(text: str):
    doc_embedder = JinaEmbeddings(
        jina_api_key=DOC_EMBEDDER, model_name=MODEL_NAME, session=_session
    )
    return await doc_embedder.aembed_documents([text])


async def embed_query(text: str):
    query_embedder = JinaEmbeddings(
        jina_api_key=QUERY_EMBEDDER, model_name=MODEL_NAME, session=_session
    )
    return await query_embedder.aembed_query(text)

