import pytest


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/admin/status"),
        ("POST", "/admin/rebuilds"),
        ("PUT", "/admin/products/board"),
        ("DELETE", "/admin/products/board"),
    ],
)
async def test_administration_requires_a_valid_token(client, product, method, path):
    for token in [None, "incorrect-token", "токен"]:
        # Заголовки HTTP передаются байтами: даже неожиданный токен не должен обрушать проверку доступа.
        headers = {
            "If-Match": "1" if method == "DELETE" else "0",
            "Idempotency-Key": "unauthorized",
        }
        if token is not None:
            headers["X-Admin-Token"] = token.encode()
        response = await client.request(
            method, path, headers=headers, json=product().model_dump() if method == "PUT" else None
        )
        assert response.status_code == 403


async def test_admin_can_create_read_edit_and_delete(client, product, admin_headers):
    response = await client.put(
        "/admin/products/board",
        headers={**admin_headers, "If-Match": "0", "Idempotency-Key": "create"},
        json=product().model_dump(),
    )
    assert response.status_code == 200
    original = response.json()
    assert (await client.get("/products/board")).json() == original
    response = await client.put(
        "/admin/products/board",
        headers={**admin_headers, "If-Match": str(original["version"]), "Idempotency-Key": "edit"},
        json=product(price_cents=6600).model_dump(),
    )
    assert response.status_code == 200
    edited = response.json()
    assert edited["price_cents"] == 6600
    response = await client.delete(
        "/admin/products/board",
        headers={**admin_headers, "If-Match": str(edited["version"]), "Idempotency-Key": "delete"},
    )
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert (await client.get("/products/board")).status_code == 404
    status = await client.get("/admin/status", headers=admin_headers)
    assert status.status_code == 200
    assert status.json()["product_count"] == 0
    assert status.json()["catalog"]["sequence"] == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"price_cents": 1.2},
        {"price_cents": True},
        {"price_cents": -1},
        {"in_stock": "true"},
        {"family": "bad family"},
        {"unknown_field": "value"},
    ],
)
async def test_product_validation_rejects_lossy_or_unexpected_values(
    client, product, admin_headers, changes
):
    response = await client.put(
        "/admin/products/board",
        headers={**admin_headers, "If-Match": "0", "Idempotency-Key": "invalid"},
        json={**product().model_dump(), **changes},
    )
    assert response.status_code == 422
    assert (await client.get("/products/board")).status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "keyboard", "min_price": 100, "max_price": 1},
        {"query": "keyboard", "limit": 51},
        {"query": "keyboard", "mode": "unknown"},
        {"query": "keyboard", "script": "arbitrary query"},
    ],
)
async def test_search_validation_rejects_invalid_requests_before_search(client, payload):
    assert (await client.post("/search", json=payload)).status_code == 422


async def test_mutations_require_version_and_idempotency_headers(client, product, admin_headers):
    for headers in [
        admin_headers,
        {**admin_headers, "If-Match": "0"},
        {**admin_headers, "Idempotency-Key": "missing-version"},
    ]:
        response = await client.put(
            "/admin/products/board", headers=headers, json=product().model_dump()
        )
        assert response.status_code == 422


async def test_sku_cannot_inject_search_paths(client, product, admin_headers):
    response = await client.put(
        "/admin/products/bad.name",
        headers={**admin_headers, "If-Match": "0", "Idempotency-Key": "invalid-sku"},
        json=product().model_dump(),
    )
    assert response.status_code == 422


async def test_health_and_metrics_are_public(client):
    assert (await client.get("/health")).json() == {"status": "ok"}
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "searchharbor_search_requests" in response.text
