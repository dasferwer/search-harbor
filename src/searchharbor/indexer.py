import asyncio
import contextlib
import hashlib
import logging

from .catalog import canonical
from .config import settings
from .db import engine, execute, one
from .models import Models, document_text
from .search_engine import MissingIndex, SearchEngine

log = logging.getLogger(__name__)


async def embedded_documents(conn, events, models):
    texts = {
        hashlib.sha256(
            (models.identity + document_text(row["body"])).encode()
        ).hexdigest(): document_text(row["body"])
        for row in events
    }
    vectors, missing = {}, []
    for key in texts:
        cached = await one(conn, "SELECT vector FROM embeddings WHERE hash=:hash", hash=key)
        if cached:
            vectors[key] = cached["vector"]
        else:
            missing.append(key)
    if missing:
        calculated = await asyncio.to_thread(models.documents, [texts[key] for key in missing])
        for key, vector in zip(missing, calculated, strict=True):
            vectors[key] = vector
            await execute(
                conn,
                "INSERT INTO embeddings(hash,vector) VALUES(:hash,CAST(:vector AS jsonb)) ON CONFLICT DO NOTHING",
                hash=key,
                vector=canonical(vector),
            )
    return [
        {
            **row["body"],
            "vector": vectors[
                hashlib.sha256((models.identity + document_text(row["body"])).encode()).hexdigest()
            ],
        }
        for row in events
    ]


async def step(search, models):
    async with engine.begin() as conn:
        locked = await one(conn, "SELECT pg_try_advisory_xact_lock(220013) AS acquired")
        if not locked["acquired"]:
            return False
        state = await one(conn, "SELECT * FROM catalog_state WHERE id=1")
        if state["model_hash"] is None:
            await execute(
                conn, "UPDATE catalog_state SET model_hash=:hash WHERE id=1", hash=models.identity
            )
        elif state["model_hash"] != models.identity:
            raise RuntimeError("Embedding model changed; do not mix incompatible vectors")
        generations = list(
            (
                await execute(
                    conn,
                    "SELECT * FROM generations WHERE status IN ('active','building') ORDER BY status",
                )
            ).mappings()
        )
        progressed = False
        for generation in generations:
            name = generation["name"]
            try:
                await search.ensure_index(name, create=generation["cursor"] == 0)
            except MissingIndex:
                if generation["status"] == "building":
                    # Сначала сохраняем нулевой курсор. Если создать индекс до commit, падение оставит пустой индекс со старым курсором.
                    await execute(
                        conn,
                        "UPDATE generations SET cursor=0,error='Missing building index; replay scheduled' WHERE name=:name",
                        name=name,
                    )
                    progressed = True
                else:
                    await execute(
                        conn,
                        "UPDATE generations SET error='Index missing; request rebuild' WHERE name=:name",
                        name=name,
                    )
                continue
            events = list(
                (
                    await execute(
                        conn,
                        "SELECT * FROM events WHERE sequence>:cursor ORDER BY sequence LIMIT :limit",
                        cursor=generation["cursor"],
                        limit=settings.batch_size,
                    )
                ).mappings()
            )
            cursor = generation["cursor"]
            if events:
                documents = await embedded_documents(conn, events, models)
                await search.bulk(name, documents)
                if settings.worker_after_bulk_delay:
                    await asyncio.sleep(settings.worker_after_bulk_delay)
                # Если worker упадёт после bulk, эта пачка придёт снова. Версия документа не даст старому событию затереть новое.
                cursor = events[-1]["sequence"]
                await execute(
                    conn,
                    "UPDATE generations SET cursor=:cursor,error=NULL WHERE name=:name",
                    cursor=cursor,
                    name=name,
                )
                progressed = True
            await execute(conn, "UPDATE generations SET error=NULL WHERE name=:name", name=name)
            if generation["status"] == "building":
                state = await one(conn, "SELECT * FROM catalog_state WHERE id=1 FOR UPDATE")
                if cursor == state["sequence"]:
                    await search.refresh(name)
                    # Указатель переключается в PostgreSQL. Запросы уже выбравшие старый индекс могут спокойно завершиться.
                    await execute(
                        conn, "UPDATE generations SET status='retired' WHERE status='active'"
                    )
                    await execute(
                        conn,
                        "UPDATE generations SET status='active',activated_at=now() WHERE name=:name",
                        name=name,
                    )
        return progressed


async def heartbeat():
    while True:
        try:
            async with engine.begin() as conn:
                await execute(
                    conn,
                    "INSERT INTO heartbeats(name) VALUES('worker') ON CONFLICT(name) DO UPDATE SET seen_at=now()",
                )
        except Exception:
            log.exception("Heartbeat failed")
        await asyncio.sleep(3)


async def run():
    models = await asyncio.to_thread(Models)
    search = SearchEngine()
    pulse = asyncio.create_task(heartbeat())
    try:
        while True:
            try:
                if not await step(search, models):
                    await asyncio.sleep(0.5)
            except Exception as error:
                log.exception("Index update failed; retrying from the saved cursor")
                async with engine.begin() as conn:
                    await execute(
                        conn,
                        "UPDATE generations SET error=:error WHERE status IN ('active','building')",
                        error=type(error).__name__,
                    )
                await asyncio.sleep(2)
    finally:
        pulse.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pulse
        await search.close()
        await engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
