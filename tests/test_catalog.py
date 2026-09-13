import asyncio

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import DBAPIError

from searchharbor.catalog import mutate, start_rebuild
from searchharbor.db import engine, execute, one


async def test_concurrent_identical_requests_commit_one_event(client, product, admin_headers):
    headers = {**admin_headers, "If-Match": "0", "Idempotency-Key": "same-create"}
    responses = await asyncio.gather(
        *[
            client.put("/admin/products/keyboard", headers=headers, json=product().model_dump())
            for _ in range(20)
        ]
    )
    assert {response.status_code for response in responses} == {200}
    assert {response.json()["version"] for response in responses} == {1}
    async with engine.connect() as conn:
        counts = await one(
            conn,
            "SELECT (SELECT count(*) FROM products) AS products,"
            "(SELECT count(*) FROM events) AS events,"
            "(SELECT count(*) FROM requests) AS requests,"
            "(SELECT sequence FROM catalog_state WHERE id=1) AS sequence",
        )
    assert dict(counts) == {"products": 1, "events": 1, "requests": 1, "sequence": 1}


async def test_same_idempotency_key_rejects_different_payload_or_operation(product):
    await mutate("board", product(), 0, "request-1")
    for sku, payload, expected, deleted in [
        ("board", product(price_cents=9000), 0, False),
        ("another", product(), 0, False),
        ("board", None, 1, True),
    ]:
        with pytest.raises(HTTPException) as error:
            await mutate(sku, payload, expected, "request-1", deleted)
        assert error.value.status_code == 409


async def test_concurrent_updates_only_one_expected_version_wins(product):
    original = await mutate("board", product(), 0, "create")
    results = await asyncio.gather(
        *[
            mutate(
                "board", product(price_cents=5000 + offset), original["version"], f"edit-{offset}"
            )
            for offset in range(15)
        ],
        return_exceptions=True,
    )
    winners = [result for result in results if isinstance(result, dict)]
    conflicts = [result for result in results if isinstance(result, HTTPException)]
    assert len(winners) == 1
    assert len(conflicts) == 14
    assert all(conflict.status_code == 409 for conflict in conflicts)
    async with engine.connect() as conn:
        assert (await one(conn, "SELECT count(*) AS n FROM events"))["n"] == 2


async def test_original_retry_does_not_restore_updated_product(product):
    original = await mutate("board", product(), 0, "create")
    edited = await mutate("board", product(price_cents=9900), original["version"], "edit")
    assert await mutate("board", product(), 0, "create") == original
    async with engine.connect() as conn:
        row = await one(conn, "SELECT body FROM products WHERE sku='board'")
    assert row["body"] == edited


async def test_delete_retry_and_recreation_require_tombstone_version(product):
    original = await mutate("board", product(), 0, "create")
    deleted = await mutate("board", None, original["version"], "delete", deleted=True)
    assert await mutate("board", None, original["version"], "delete", deleted=True) == deleted
    with pytest.raises(HTTPException) as error:
        await mutate("board", product(), 0, "incorrect-recreate")
    assert error.value.status_code == 409
    recreated = await mutate("board", product(price_cents=6000), deleted["version"], "recreate")
    assert recreated["version"] > deleted["version"]
    assert recreated["deleted"] is False


async def test_delete_missing_product_is_not_an_event():
    with pytest.raises(HTTPException) as error:
        await mutate("absent", None, 0, "delete-missing", deleted=True)
    assert error.value.status_code == 404
    async with engine.connect() as conn:
        assert (await one(conn, "SELECT sequence FROM catalog_state WHERE id=1"))["sequence"] == 0


@pytest.mark.parametrize("statement", ["UPDATE events SET sku='changed'", "DELETE FROM events"])
async def test_event_history_is_immutable(product, statement):
    saved = await mutate("board", product(), 0, "create")
    with pytest.raises(DBAPIError, match="append-only"):
        async with engine.begin() as conn:
            await execute(conn, statement)
    async with engine.connect() as conn:
        events = list((await execute(conn, "SELECT body FROM events")).scalars())
    assert events == [saved]


async def test_two_concurrent_rebuild_requests_create_only_one_generation():
    results = await asyncio.gather(start_rebuild(), start_rebuild(), return_exceptions=True)
    assert sum(isinstance(item, HTTPException) and item.status_code == 409 for item in results) == 1
    async with engine.connect() as conn:
        assert (await one(conn, "SELECT count(*) AS n FROM generations"))["n"] == 1


async def test_catalog_and_event_roll_back_when_request_cannot_be_saved(product):
    async with engine.begin() as conn:
        await execute(
            conn, "ALTER TABLE requests ADD CONSTRAINT reject_test_key CHECK(key <> 'fail')"
        )
    try:
        with pytest.raises(DBAPIError):
            await mutate("board", product(), 0, "fail")
        async with engine.connect() as conn:
            assert (await one(conn, "SELECT sequence FROM catalog_state WHERE id=1"))[
                "sequence"
            ] == 0
            assert (await one(conn, "SELECT count(*) AS n FROM products"))["n"] == 0
            assert (await one(conn, "SELECT count(*) AS n FROM events"))["n"] == 0
    finally:
        async with engine.begin() as conn:
            await execute(conn, "ALTER TABLE requests DROP CONSTRAINT reject_test_key")
