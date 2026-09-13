import asyncio

import pytest

from searchharbor.catalog import mutate, start_rebuild
from searchharbor.config import settings
from searchharbor.db import engine, one
from searchharbor.indexer import step
from searchharbor.models import document_text
from searchharbor.retrieval import retrieve
from searchharbor.schemas import Search


async def active_generation():
    async with engine.connect() as conn:
        return await one(conn, "SELECT * FROM generations WHERE status='active'")


class DelegatingSearch:
    def __init__(self, search):
        self.search = search

    def __getattr__(self, name):
        return getattr(self.search, name)


class CrashAfterBulk(DelegatingSearch):
    async def bulk(self, name, documents):
        await self.search.bulk(name, documents)
        # Индекс уже принял пачку, но транзакция с курсором ещё не успела завершиться.
        raise RuntimeError("simulated worker crash after bulk")


class PauseBulk(DelegatingSearch):
    def __init__(self, search, name=None):
        super().__init__(search)
        self.name = name
        self.entered = asyncio.Event()
        self.resume = asyncio.Event()

    async def bulk(self, name, documents):
        await self.search.bulk(name, documents)
        if self.name is None or name == self.name:
            self.entered.set()
            await asyncio.wait_for(self.resume.wait(), timeout=15)


async def test_bulk_committed_before_crash_replays_from_durable_cursor(search, models, product):
    saved = await mutate("board", product(), 0, "create")
    generation = await start_rebuild()
    with pytest.raises(RuntimeError, match="simulated worker crash"):
        await step(CrashAfterBulk(search), models)
    indexed = await search.request("GET", f"/{generation['name']}/_doc/board")
    assert indexed["_source"]["version"] == saved["version"]
    async with engine.connect() as conn:
        row = await one(conn, "SELECT * FROM generations WHERE name=:name", name=generation["name"])
        assert row["cursor"] == 0
        assert row["status"] == "building"
        assert (await one(conn, "SELECT model_hash FROM catalog_state WHERE id=1"))[
            "model_hash"
        ] is None
    assert await step(search, models)
    await search.refresh(generation["name"])
    assert (await search.request("GET", f"/{generation['name']}/_count"))["count"] == 1
    row = await active_generation()
    assert row["cursor"] == saved["version"]


async def test_old_document_cannot_overwrite_a_durable_tombstone(search, models, product):
    original = await mutate("board", product(), 0, "create")
    generation = await start_rebuild()
    await step(search, models)
    deleted = await mutate("board", None, original["version"], "delete", deleted=True)
    await step(search, models)
    await search.bulk(
        generation["name"],
        [{**original, "vector": models.documents([document_text(original)])[0]}],
    )
    await search.refresh(generation["name"])
    source = (await search.request("GET", f"/{generation['name']}/_doc/board"))["_source"]
    assert source["version"] == deleted["version"]
    assert source["deleted"] is True
    assert not (await retrieve(Search(query="keyboard", mode="lexical"), search, models))["items"]


async def test_equal_version_replay_is_idempotent(search, models, product):
    saved = await mutate("board", product(), 0, "create")
    generation = await start_rebuild()
    await step(search, models)
    document = {**saved, "vector": models.documents([document_text(saved)])[0]}
    await search.bulk(generation["name"], [document, document, document])
    await search.refresh(generation["name"])
    assert (await search.request("GET", f"/{generation['name']}/_count"))["count"] == 1
    assert (await search.request("GET", f"/{generation['name']}/_doc/board"))["_version"] == 1


async def test_competing_worker_does_not_process_the_same_batch(search, models, product):
    await mutate("board", product(), 0, "create")
    await start_rebuild()
    paused = PauseBulk(search)
    first = asyncio.create_task(step(paused, models))
    try:
        await asyncio.wait_for(paused.entered.wait(), timeout=10)
        assert await asyncio.wait_for(step(search, models), timeout=3) is False
    finally:
        paused.resume.set()
        await first
    assert (await active_generation())["cursor"] == 1


async def test_rebuild_keeps_live_edits_and_deletions(search, models, product):
    first = await mutate("board", product(), 0, "create-board")
    second = await mutate(
        "mouse", product(title="Wireless mouse", family="mouse"), 0, "create-mouse"
    )
    original = await start_rebuild()
    await step(search, models)
    replacement = await start_rebuild()
    paused = PauseBulk(search, replacement["name"])
    rebuilding = asyncio.create_task(step(paused, models))
    try:
        await asyncio.wait_for(paused.entered.wait(), timeout=10)
        edited = await mutate("board", product(price_cents=7700), first["version"], "edit-board")
        deleted = await mutate("mouse", None, second["version"], "delete-mouse", deleted=True)
        assert (await active_generation())["name"] == original["name"]
    finally:
        paused.resume.set()
        await rebuilding
    assert (await active_generation())["name"] == original["name"]
    await step(search, models)
    active = await active_generation()
    assert active["name"] == replacement["name"]
    assert active["cursor"] == deleted["version"]
    board = (await search.request("GET", f"/{active['name']}/_doc/board"))["_source"]
    mouse = (await search.request("GET", f"/{active['name']}/_doc/mouse"))["_source"]
    assert board["version"] == edited["version"]
    assert board["price_cents"] == 7700
    assert mouse["deleted"] is True
    async with engine.connect() as conn:
        assert (
            await one(
                conn, "SELECT status FROM generations WHERE name=:name", name=original["name"]
            )
        )["status"] == "retired"


async def test_rebuild_is_not_exposed_until_all_batches_are_indexed(
    search, models, product, monkeypatch
):
    monkeypatch.setattr(settings, "batch_size", 1)
    await mutate("first", product(), 0, "first")
    await mutate("second", product(), 0, "second")
    await start_rebuild()
    assert await step(search, models)
    assert await active_generation() is None
    assert await step(search, models)
    assert (await active_generation())["cursor"] == 2


async def test_empty_catalog_can_become_ready(search, models):
    generation = await start_rebuild()
    assert await step(search, models) is False
    assert (await active_generation())["name"] == generation["name"]
    result = await retrieve(Search(query="keyboard"), search, models)
    assert result["items"] == []
    assert result["lag_events"] == 0


async def test_model_identity_change_stops_indexing_before_cursor_moves(search, models, product):
    first = await mutate("board", product(), 0, "create")
    generation = await start_rebuild()
    await step(search, models)
    await mutate("board", product(price_cents=9900), first["version"], "edit")
    models.identity = "incompatible-model"
    with pytest.raises(RuntimeError, match="Embedding model changed"):
        await step(search, models)
    assert (await active_generation())["cursor"] == first["version"]
    indexed = await search.request("GET", f"/{generation['name']}/_doc/board")
    assert indexed["_source"]["version"] == first["version"]


async def test_identical_text_reuses_cached_embedding(search, models, product):
    calls = []
    original = models.documents

    def record(texts):
        calls.extend(texts)
        return original(texts)

    models.documents = record
    first = await mutate("board", product(), 0, "create")
    await start_rebuild()
    await step(search, models)
    await mutate("board", product(price_cents=9900), first["version"], "edit-price")
    await step(search, models)
    await start_rebuild()
    await step(search, models)
    assert len(calls) == 1
    async with engine.connect() as conn:
        assert (await one(conn, "SELECT count(*) AS n FROM embeddings"))["n"] == 1


async def test_failed_bulk_does_not_advance_saved_cursor(search, models, product):
    await mutate("board", product(), 0, "create")
    generation = await start_rebuild()

    class RejectBulk(DelegatingSearch):
        async def bulk(self, name, documents):
            raise RuntimeError("OpenSearch temporarily unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        await step(RejectBulk(search), models)
    async with engine.connect() as conn:
        assert (
            await one(
                conn, "SELECT cursor FROM generations WHERE name=:name", name=generation["name"]
            )
        )["cursor"] == 0
        assert (await one(conn, "SELECT count(*) AS n FROM events"))["n"] == 1
    await step(search, models)
    assert (await active_generation())["cursor"] == 1


async def test_partial_bulk_replays_successful_and_rejected_items(search, models, product):
    await mutate("first", product(), 0, "first")
    await mutate("second", product(), 0, "second")
    generation = await start_rebuild()

    class InvalidSecondVector(DelegatingSearch):
        async def bulk(self, name, documents):
            # Первая запись применится, а вторая получит ошибку размерности в том же bulk.
            await self.search.bulk(name, [documents[0], {**documents[1], "vector": [1.0]}])

    with pytest.raises(RuntimeError, match="Bulk indexing rejected 1 documents"):
        await step(InvalidSecondVector(search), models)
    assert (await search.request("GET", f"/{generation['name']}/_doc/first"))["found"]
    assert (await search.client.get(f"/{generation['name']}/_doc/second")).status_code == 404
    async with engine.connect() as conn:
        assert (
            await one(
                conn, "SELECT cursor FROM generations WHERE name=:name", name=generation["name"]
            )
        )["cursor"] == 0
    await step(search, models)
    assert (await active_generation())["cursor"] == 2
    assert (await search.request("GET", f"/{generation['name']}/_count"))["count"] == 2


async def test_lost_active_index_requires_rebuild_from_event_history(
    client, search, models, product
):
    saved = await mutate("board", product(), 0, "create")
    original = await start_rebuild()
    await step(search, models)
    assert (await client.get("/ready")).status_code == 200
    await search.request("DELETE", "/" + original["name"])
    assert (await client.get("/ready")).status_code == 503
    assert await step(search, models) is False
    # Курсор уже прошёл событие. Пустой индекс с прежним именем выглядел бы готовым и терял товары.
    assert (await search.client.head("/" + original["name"])).status_code == 404
    active = await active_generation()
    assert active["cursor"] == saved["version"]
    assert active["error"] is not None
    replacement = await start_rebuild()
    assert await step(search, models)
    assert (await active_generation())["name"] == replacement["name"]
    assert (await client.get("/ready")).status_code == 200
    result = await retrieve(Search(query="keyboard", mode="lexical"), search, models)
    assert [item["product"] for item in result["items"]] == [saved]


async def test_lost_building_index_commits_cursor_reset_before_replaying(
    search, models, product, monkeypatch
):
    monkeypatch.setattr(settings, "batch_size", 1)
    first = await mutate("first", product(), 0, "first")
    second = await mutate("second", product(price_cents=6500), 0, "second")
    generation = await start_rebuild()
    name = generation["name"]
    assert await step(search, models)
    async with engine.connect() as conn:
        row = await one(conn, "SELECT * FROM generations WHERE name=:name", name=name)
    assert row["cursor"] == first["version"]
    assert row["status"] == "building"
    await search.request("DELETE", "/" + name)
    assert await step(search, models)
    async with engine.connect() as conn:
        row = await one(conn, "SELECT * FROM generations WHERE name=:name", name=name)
    assert row["cursor"] == 0
    assert row["status"] == "building"
    assert row["error"] is not None
    # Между сбросом курсора и созданием индекса можно перезапустить worker без потери первой пачки.
    assert (await search.client.head("/" + name)).status_code == 404
    assert await step(search, models)
    assert await active_generation() is None
    assert await step(search, models)
    active = await active_generation()
    assert active["name"] == name
    assert active["cursor"] == second["version"]
    assert active["error"] is None
    result = await retrieve(Search(query="keyboard", mode="lexical"), search, models)
    assert {item["product"]["sku"]: item["product"] for item in result["items"]} == {
        "first": first,
        "second": second,
    }
