import hashlib
import json
from uuid import uuid4

from fastapi import HTTPException

from .db import engine, execute, one


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


async def mutate(sku, product, expected, key, deleted=False):
    payload = product.model_dump() if product else None
    body_hash = hashlib.sha256(
        canonical(
            {"sku": sku, "product": payload, "expected": expected, "deleted": deleted}
        ).encode()
    ).hexdigest()
    async with engine.begin() as conn:
        state = await one(conn, "SELECT * FROM catalog_state WHERE id=1 FOR UPDATE")
        previous = await one(conn, "SELECT * FROM requests WHERE key=:key", key=key)
        if previous:
            if previous["body_hash"] != body_hash:
                raise HTTPException(409, "Idempotency key was used with a different request")
            return previous["result"]
        row = await one(conn, "SELECT * FROM products WHERE sku=:sku", sku=sku)
        if (row["version"] if row else 0) != expected:
            raise HTTPException(409, "Product version changed; read it again before updating")
        if deleted and (row is None or row["deleted"]):
            raise HTTPException(404, "Product not found")
        seq = state["sequence"] + 1
        body = {
            **(row["body"] if deleted else payload),
            "sku": sku,
            "version": seq,
            "deleted": deleted,
        }
        await execute(conn, "UPDATE catalog_state SET sequence=:seq WHERE id=1", seq=seq)
        await execute(
            conn,
            "INSERT INTO products(sku,version,body,deleted) VALUES(:sku,:seq,CAST(:body AS jsonb),:deleted) ON CONFLICT(sku) DO UPDATE SET version=excluded.version,body=excluded.body,deleted=excluded.deleted",
            sku=sku,
            seq=seq,
            body=canonical(body),
            deleted=deleted,
        )
        # Каталог и запись для индекса коммитятся вместе, даже если OpenSearch сейчас недоступен.
        await execute(
            conn,
            "INSERT INTO events(sequence,sku,body) VALUES(:seq,:sku,CAST(:body AS jsonb))",
            seq=seq,
            sku=sku,
            body=canonical(body),
        )
        await execute(
            conn,
            "INSERT INTO requests(key,body_hash,result) VALUES(:key,:hash,CAST(:body AS jsonb))",
            key=key,
            hash=body_hash,
            body=canonical(body),
        )
        return body


async def start_rebuild():
    async with engine.begin() as conn:
        await one(conn, "SELECT id FROM catalog_state WHERE id=1 FOR UPDATE")
        if await one(conn, "SELECT name FROM generations WHERE status='building'"):
            raise HTTPException(409, "A rebuild is already running")
        count = await one(conn, "SELECT count(*) AS n FROM generations")
        if count["n"] >= 10:
            raise HTTPException(
                409, "Ten generations retained; archive old indexes before another rebuild"
            )
        name = "catalog-" + uuid4().hex
        return await one(
            conn,
            "INSERT INTO generations(name,status) VALUES(:name,'building') RETURNING *",
            name=name,
        )
