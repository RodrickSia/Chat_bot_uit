import requests
from langchain_community.embeddings import JinaEmbeddings
from functools import lru_cache
from pydantic import SecretStr
from typing import List


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
    async def embed_document(self, text: str) -> List[float]:
        res = await self.embedder.aembed_documents([text])
        return res[0]
    
    
    async def embed_query(self, text: str) -> List[float]:
        return await self.embedder.aembed_query(text)
