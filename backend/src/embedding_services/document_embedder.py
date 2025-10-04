# ...existing code...
import requests
from langchain_community.embeddings import JinaEmbeddings
from functools import lru_cache
import asyncio
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Any



class Settings(BaseSettings):
    
    # BaseSetting autoload from env vars
    api_document_embedding: SecretStr = SecretStr("") # This is to make pylance stfu about missing constructor arg  
    api_query_embedding: SecretStr = SecretStr("")
    model_name: str = "jina-embeddings-v4"
    
    # env_file is path to the env file it does not look for env in parent dirs
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

@lru_cache
def get_settings() -> Settings:
    return Settings()


class EmbeddingClient():
    def __init__(
        self,
        api_key: SecretStr,
        model_name: str = "jina-embeddings-v4",
    ):
        self.__api_key = api_key
        self.model_name = model_name
        self.session = requests.Session()
        self.embedder = JinaEmbeddings(
            jina_api_key=self.__api_key, model_name=self.model_name, session=self.session
        )

    def close(self) -> None:
        self.session.close()
    
    
    # aembed is truly async function in Langchain
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.embedder.aembed_documents(texts)
    
    async def embed_query(self, text: str) -> List[float]:
        return await self.embedder.aembed_query(text)
