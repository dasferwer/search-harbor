import pytest

from searchharbor.catalog import mutate, start_rebuild
from searchharbor.indexer import step
from searchharbor.models import document_text
from searchharbor.retrieval import retrieve
from searchharbor.schemas import Search


async def make_ready(search, models):
    generation = await start_rebuild()
    await step(search, models)
    await search.refresh(generation["name"])
    return generation["name"]


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid", "rerank"])
async def test_all_search_paths_return_indexed_catalog_products(
    client, product, search, models, mode
):
    saved = await mutate("board", product(), 0, "create")
    await make_ready(search, models)
    response = await client.post("/search", json={"query": "wireless keyboard", "mode": mode})
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == mode
    assert result["items"][0]["product"] == saved
    assert result["lag_events"] == 0
    assert set(result["items"][0]["ranks"]) == (
        {"lexical"}
        if mode == "lexical"
        else {"semantic"}
        if mode == "semantic"
        else {"lexical", "semantic"}
    )
    if mode == "rerank":
        assert result["items"][0]["rrf_score"] > 0


async def test_lexical_search_recovers_a_typo(search, models, product):
    await mutate("board", product(), 0, "create")
    await make_ready(search, models)
    result = await retrieve(Search(query="keybord", mode="lexical"), search, models)
    assert [item["product"]["sku"] for item in result["items"]] == ["board"]


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid", "rerank"])
async def test_filters_apply_to_every_retrieval_mode(search, models, product, mode):
    cases = {
        "included": {},
        "cheap": {"price_cents": 100},
        "expensive": {"price_cents": 9000},
        "other-category": {"category": "accessories"},
        "other-brand": {"brand": "other"},
        "sold-out": {"in_stock": False},
    }
    for sku, changes in cases.items():
        await mutate(sku, product(**changes), 0, sku)
    await make_ready(search, models)
    request = Search(
        query="keyboard",
        mode=mode,
        category="keyboards",
        brand="harbor",
        min_price=4000,
        max_price=5000,
    )
    result = await retrieve(request, search, models)
    assert [item["product"]["sku"] for item in result["items"]] == ["included"]
    result = await retrieve(request.model_copy(update={"in_stock": False}), search, models)
    assert {item["product"]["sku"] for item in result["items"]} == {"included", "sold-out"}


async def test_updated_price_is_hidden_until_index_reaches_current_version(search, models, product):
    first = await mutate("board", product(price_cents=1000), 0, "create")
    name = await make_ready(search, models)
    updated = await mutate("board", product(price_cents=9000), first["version"], "price-edit")
    result = await retrieve(Search(query="keyboard", mode="lexical"), search, models)
    assert result["items"] == []
    assert result["lag_events"] == 1
    await step(search, models)
    await search.refresh(name)
    result = await retrieve(Search(query="keyboard", mode="lexical"), search, models)
    assert result["items"][0]["product"] == updated
    assert result["lag_events"] == 0


async def test_deleted_product_disappears_before_worker_handles_tombstone(
    client, search, models, product
):
    first = await mutate("board", product(), 0, "create")
    await make_ready(search, models)
    await mutate("board", None, first["version"], "delete", deleted=True)
    result = await retrieve(Search(query="keyboard", mode="lexical"), search, models)
    assert result["items"] == []
    assert (await client.get("/products/board")).status_code == 404


async def test_index_only_document_is_filtered_against_postgres(search, models, product):
    saved = await mutate("board", product(), 0, "create")
    name = await make_ready(search, models)
    ghost = {**saved, "sku": "ghost", "vector": models.documents([document_text(saved)])[0]}
    await search.bulk(name, [ghost])
    await search.refresh(name)
    result = await retrieve(Search(query="keyboard", mode="lexical"), search, models)
    assert [item["product"]["sku"] for item in result["items"]] == ["board"]


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid", "rerank"])
async def test_distinct_families_collapse_product_variants(search, models, product, mode):
    for sku, family in [
        ("black", "quiet-board"),
        ("white", "quiet-board"),
        ("compact", "travel-board"),
    ]:
        await mutate(sku, product(family=family), 0, sku)
    await make_ready(search, models)
    result = await retrieve(
        Search(query="keyboard", mode=mode, distinct_families=True), search, models
    )
    assert len(result["items"]) == 2
    assert {item["product"]["family"] for item in result["items"]} == {
        "quiet-board",
        "travel-board",
    }


async def test_query_model_mismatch_returns_unavailable(client, search, models, product):
    await mutate("board", product(), 0, "create")
    await make_ready(search, models)
    models.identity = "another-model"
    response = await client.post("/search", json={"query": "keyboard"})
    assert response.status_code == 503
    assert "model" in response.json()["detail"].lower()
    assert (await client.get("/ready")).status_code == 503


async def test_unbuilt_catalog_returns_clear_readiness_error(client):
    assert (await client.get("/health")).status_code == 200
    assert (await client.get("/ready")).status_code == 503
    assert (await client.post("/search", json={"query": "keyboard"})).status_code == 503


async def test_backend_failure_returns_503_without_partial_results(
    client, search, models, product, monkeypatch
):
    await mutate("board", product(), 0, "create")
    await make_ready(search, models)

    async def unavailable(*args, **kwargs):
        raise RuntimeError("Search did not complete on all shards")

    monkeypatch.setattr(search, "search", unavailable)
    response = await client.post("/search", json={"query": "keyboard"})
    assert response.status_code == 503
    assert "items" not in response.json()
