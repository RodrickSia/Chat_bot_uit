from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from functools import lru_cache
from pydantic import SecretStr


ENV_PATH = Path(__file__).parent.parent.parent / ".env" 
class Settings(BaseSettings):
    
    # BaseSetting autoload from env vars
    api_document_embedding: SecretStr = SecretStr("")  # This is to make pylance stfu about missing constructor arg
    api_query_embedding: SecretStr = SecretStr("")
    model_name: str = "jina-embeddings-v4"
    host_address: str = "0.0.0.0"
    # env_file is path to the env file it does not look for env in parent dirs
    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding="utf-8")
    
@lru_cache
def get_settings() -> Settings:
    return Settings()



def test():
    print(ENV_PATH)
    settings = get_settings()
    print(settings.api_document_embedding.get_secret_value())
    print(settings.api_query_embedding.get_secret_value())
    print(settings.model_name)
    print(settings.host_address)
    
if __name__ == "__main__":
    test()