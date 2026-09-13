import hashlib
import math
import re
from urllib.parse import urlparse

import httpx
import pytest
from sqlalchemy.engine import make_url

from searchharbor.config import MODEL_DIM, settings
from searchharbor.db import engine, execute
from searchharbor.main import app
from searchharbor.schemas import Product
from searchharbor.search_engine import SearchEngine


class FakeModels:
    identity = "test-model"

    @staticmethod
    def vector(text):
        vector = [0.0] * MODEL_DIM
        for token in re.findall(r"\w+", text.lower()):
            position = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4]) % MODEL_DIM
            vector[position] += 1.0
        if not any(vector):
            vector[0] = 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    def documents(self, texts):
        return [self.vector(text) for text in texts]

    def query(self, text):
        return self.vector(text)

    def rerank(self, query, texts):
        terms = set(query.lower().split())
        return [float(len(terms & set(text.lower().split()))) for text in texts]


@pytest.fixture
def models():
    return FakeModels()


@pytest.fixture
def product():
    def make(**changes):
        return Product(
            **{
                "title": "Wireless keyboard",
                "description": "A quiet wireless keyboard for a comfortable office desk.",
                "category": "keyboards",
                "brand": "harbor",
                "family": "quiet-board",
                "price_cents": 4500,
                "in_stock": True,
                **changes,
            }
        )

    return make


@pytest.fixture
async def search():
    db_url = make_url(settings.database_url)
    search_url = urlparse(settings.opensearch_url)
    # Эта проверка стоит до первого запроса: тесты не должны очищать демонстрационный каталог.
    if not (
        settings.testing
        and db_url.host == "test-db"
        and db_url.database == "search"
        and search_url.hostname == "test-search"
        and search_url.port == 9200
    ):
        pytest.fail("Destructive tests require TESTING=true and isolated test-db/test-search hosts")
    instance = SearchEngine()
    yield instance
    await instance.close()


@pytest.fixture(autouse=True)
async def clean_state(search):
    response = await search.client.get("/_cat/indices", params={"format": "json"})
    response.raise_for_status()
    for row in response.json():
        name = row["index"]
        if name.startswith("catalog-"):
            await search.request("DELETE", "/" + name)
    async with engine.begin() as conn:
        await execute(
            conn,
            "TRUNCATE products,events,requests,generations,embeddings,heartbeats,catalog_state",
        )
        await execute(conn, "INSERT INTO catalog_state(id) VALUES(1)")
    yield


@pytest.fixture
async def client(search, models):
    # ASGITransport не запускает lifespan; вместо настоящих моделей подставляем небольшую заглушку.
    app.state.models = models
    app.state.search = search
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as session:
        yield session


@pytest.fixture
def admin_headers():
    return {"X-Admin-Token": settings.admin_token}


@pytest.fixture(scope="session", autouse=True)
async def close_database_pool():
    yield
    await engine.dispose()
