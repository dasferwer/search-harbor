from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_DIM = 384


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+asyncpg://search:search@database:5432/search"
    opensearch_url: str = "http://opensearch:9200"
    admin_token: str = Field(
        default="local-demo-searchharbor-admin-change-before-deployment", min_length=32
    )
    model_cache: str = "/models"
    batch_size: int = Field(default=100, ge=1, le=500)
    worker_after_bulk_delay: float = Field(default=0, ge=0, le=30)
    testing: bool = False


settings = Settings()
