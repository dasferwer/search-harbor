import asyncio

from fastapi import HTTPException

from .db import engine, execute, one
from .models import document_text


def fuse(rankings):
    candidates = {}
    for label, hits in rankings.items():
        for rank, hit in enumerate(hits, 1):
            sku = hit["_id"]
            item = candidates.setdefault(
                sku, {"product": hit["_source"], "score": 0.0, "ranks": {}}
            )
            item["score"] += 1 / (60 + rank)
            item["ranks"][label] = rank
    return sorted(candidates.values(), key=lambda item: (-item["score"], item["product"]["sku"]))


def allowed(product, request):
    return (
        not product["deleted"]
        and request.min_price <= product["price_cents"] <= request.max_price
        and (not request.in_stock or product["in_stock"])
        and (request.category is None or product["category"] == request.category)
        and (request.brand is None or product["brand"] == request.brand)
    )


async def retrieve(request, search, models):
    async with engine.connect() as conn:
        generation = await one(conn, "SELECT * FROM generations WHERE status='active'")
        state = await one(conn, "SELECT * FROM catalog_state WHERE id=1")
    if generation is None:
        raise HTTPException(503, "Initial index is still building")
    if state["model_hash"] != models.identity:
        raise HTTPException(503, "Query model does not match the index model")
    name, rankings = generation["name"], {}
    if request.mode != "semantic":
        rankings["lexical"] = await search.search(name, request)
    if request.mode != "lexical":
        vector = await asyncio.to_thread(models.query, request.query)
        rankings["semantic"] = await search.search(name, request, vector)
    candidates = fuse(rankings)
    skus = [item["product"]["sku"] for item in candidates]
    async with engine.connect() as conn:
        rows = (
            (
                await execute(
                    conn, "SELECT sku,version,body FROM products WHERE sku=ANY(:skus)", skus=skus
                )
            ).mappings()
            if skus
            else []
        )
        current = {row["sku"]: row for row in rows}
    # Индекс обновляется не мгновенно. Удалённый или изменённый товар лучше временно скрыть, чем показать старую цену.
    candidates = [
        item
        for item in candidates
        if item["product"]["sku"] in current
        and current[item["product"]["sku"]]["version"] == item["product"]["version"]
        and allowed(current[item["product"]["sku"]]["body"], request)
    ]
    if request.distinct_families:
        seen, distinct = set(), []
        for item in candidates:
            family = item["product"]["family"]
            if family not in seen:
                distinct.append(item)
                seen.add(family)
        candidates = distinct
    if request.mode == "rerank" and candidates:
        candidates = candidates[:60]
        texts = list(dict.fromkeys(document_text(item["product"]) for item in candidates))
        scores = await asyncio.to_thread(models.rerank, request.query, texts)
        by_text = dict(zip(texts, scores, strict=True))
        for item in candidates:
            item["rrf_score"] = item["score"]
            item["score"] = by_text[document_text(item["product"])]
        candidates.sort(key=lambda item: (-item["score"], item["product"]["sku"]))
    return {
        "query": request.query,
        "mode": request.mode,
        "index": name,
        "indexed_sequence": generation["cursor"],
        "catalog_sequence": state["sequence"],
        "lag_events": max(0, state["sequence"] - generation["cursor"]),
        "items": candidates[: request.limit],
        "candidate_count": len(candidates),
    }
