"""Проверить API, настоящие модели и путь изменения товара."""

import json
import os
from uuid import uuid4

from demo_client import DemoClient, sample_product, wait_for


def main():
    client = DemoClient(os.getenv("BASE_URL", "http://localhost:8000"))
    sku = "smoke-" + uuid4().hex
    try:
        wait_for(
            lambda: client.http.get("/ready").status_code == 200,
            timeout=300,
            description="initial index",
        )
        wait_for(client.caught_up)
        assert client.http.get("/openapi.json").status_code == 200
        assert (
            client.http.post("/admin/rebuilds", headers={"X-Admin-Token": "wrong"}).status_code
            == 403
        )
        for mode in ("lexical", "semantic", "hybrid", "rerank"):
            result = client.search("comfortable typing at a desk", mode, distinct_families=True)
            assert result["items"], mode
            assert len({item["product"]["family"] for item in result["items"]}) == len(
                result["items"]
            )
        body, key = sample_product(), uuid4().hex
        created = client.put(sku, body, key=key)
        assert client.put(sku, body, key=key) == created
        conflict = client.http.put(
            f"/admin/products/{sku}",
            json={**body, "price_cents": 7600},
            headers={"If-Match": "0", "Idempotency-Key": key},
        )
        assert conflict.status_code == 409
        wait_for(client.caught_up)
        wait_for(
            lambda: any(
                item["product"]["sku"] == sku
                for item in client.search("Recovery keyboard", category="test")["items"]
            ),
            description="created product",
        )
        updated = client.put(sku, {**body, "price_cents": 9900}, created["version"])
        # Даже до обработки события старая цена уже не должна проходить фильтр.
        assert not client.search("Recovery keyboard", category="test", max_price=8000)["items"]
        client.delete(sku, updated["version"])
        assert client.http.get(f"/products/{sku}").status_code == 404
        assert not client.search("Recovery keyboard", category="test")["items"]
        wait_for(client.caught_up)
        metrics = client.http.get("/metrics")
        assert metrics.status_code == 200 and "searchharbor_search_seconds" in metrics.text
        print(
            json.dumps(
                {
                    "result": "passed",
                    "modes": 4,
                    "idempotency": True,
                    "stale_price_hidden": True,
                    "deleted_product_hidden": True,
                }
            )
        )
    finally:
        client.clean([sku])
        client.http.close()


if __name__ == "__main__":
    main()
