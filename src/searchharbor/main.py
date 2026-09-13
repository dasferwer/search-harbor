import asyncio
import secrets
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Path, Response
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .catalog import mutate, start_rebuild
from .config import settings
from .db import engine, execute, one
from .models import Models
from .retrieval import retrieve
from .schemas import Product, Search
from .search_engine import SearchEngine

security = APIKeyHeader(name="X-Admin-Token", auto_error=False)
REQUESTS = Counter("searchharbor_search_requests_total", "Search requests", ["mode", "status"])
LATENCY = Histogram("searchharbor_search_seconds", "Search duration", ["mode"])


async def admin(token: str | None = Depends(security)):
    if not secrets.compare_digest((token or "").encode(), settings.admin_token.encode()):
        raise HTTPException(403, "Administrator token required")


@asynccontextmanager
async def lifespan(app):
    app.state.models = await asyncio.to_thread(Models)
    app.state.search = SearchEngine()
    yield
    await app.state.search.close()
    await engine.dispose()


app = FastAPI(
    title="SearchHarbor",
    version="0.1.0",
    lifespan=lifespan,
    description="English catalog search: BM25, BGE embeddings, reciprocal rank fusion, MiniLM reranking and recoverable index rebuilds.",
)


@app.get("/health", tags=["Operations"])
async def health():
    async with engine.connect() as conn:
        await execute(conn, "SELECT 1")
    return {"status": "ok"}


@app.get("/ready", tags=["Operations"])
async def ready():
    async with engine.connect() as conn:
        row = await one(conn, "SELECT name FROM generations WHERE status='active'")
        state = await one(conn, "SELECT model_hash FROM catalog_state WHERE id=1")
    if row is None:
        raise HTTPException(503, "Initial index is building")
    if state["model_hash"] != app.state.models.identity:
        raise HTTPException(503, "Embedding model mismatch")
    try:
        result = await app.state.search.request("GET", f"/{row['name']}/_count")
        if result.get("_shards", {}).get("failed", 0):
            raise RuntimeError("Count failed on a shard")
    except (httpx.HTTPError, RuntimeError):
        raise HTTPException(503, "Search backend unavailable") from None
    return {"status": "ready", "index": row["name"]}


@app.post("/search", tags=["Search"])
async def search(request: Search):
    start, status = time.monotonic(), "ok"
    try:
        return await retrieve(request, app.state.search, app.state.models)
    except HTTPException as error:
        status = "unavailable" if error.status_code == 503 else "error"
        raise
    except (httpx.HTTPError, RuntimeError):
        status = "unavailable"
        raise HTTPException(503, "Search backend unavailable; retry later") from None
    finally:
        REQUESTS.labels(request.mode, status).inc()
        LATENCY.labels(request.mode).observe(time.monotonic() - start)


@app.get("/products/{sku}", tags=["Catalog"])
async def get_product(sku: str = Path(pattern=r"^[a-zA-Z0-9_-]{1,80}$")):
    async with engine.connect() as conn:
        row = await one(conn, "SELECT body FROM products WHERE sku=:sku AND NOT deleted", sku=sku)
    if row is None:
        raise HTTPException(404, "Product not found")
    return row["body"]


@app.put("/admin/products/{sku}", dependencies=[Depends(admin)], tags=["Administration"])
async def put_product(
    product: Product,
    sku: str = Path(pattern=r"^[a-zA-Z0-9_-]{1,80}$"),
    if_match: int = Header(ge=0),
    idempotency_key: str = Header(min_length=1, max_length=128),
):
    return await mutate(sku, product, if_match, idempotency_key)


@app.delete("/admin/products/{sku}", dependencies=[Depends(admin)], tags=["Administration"])
async def delete_product(
    sku: str = Path(pattern=r"^[a-zA-Z0-9_-]{1,80}$"),
    if_match: int = Header(ge=1),
    idempotency_key: str = Header(min_length=1, max_length=128),
):
    return await mutate(sku, None, if_match, idempotency_key, deleted=True)


@app.post(
    "/admin/rebuilds", status_code=202, dependencies=[Depends(admin)], tags=["Administration"]
)
async def rebuild():
    return await start_rebuild()


@app.get("/admin/status", dependencies=[Depends(admin)], tags=["Operations"])
async def status():
    async with engine.connect() as conn:
        state = await one(conn, "SELECT * FROM catalog_state WHERE id=1")
        generations = list(
            (await execute(conn, "SELECT * FROM generations ORDER BY created_at")).mappings()
        )
        heartbeat = await one(
            conn,
            "SELECT seen_at,seen_at>now()-interval '20 seconds' AS alive FROM heartbeats WHERE name='worker'",
        )
        count = await one(conn, "SELECT count(*) AS count FROM products WHERE NOT deleted")
    return {
        "catalog": state,
        "product_count": count["count"],
        "generations": generations,
        "worker": heartbeat,
    }


@app.get("/metrics", tags=["Operations"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
