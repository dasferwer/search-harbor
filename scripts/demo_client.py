"""Небольшой клиент для проверок локального стенда."""

import time
from uuid import uuid4

import httpx

from searchharbor.config import settings

TOKEN = settings.admin_token


def wait_for(check, *, timeout=180, description="condition"):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            if value := check():
                return value
        except (httpx.HTTPError, KeyError) as error:
            last_error = str(error)
        time.sleep(0.3)
    raise TimeoutError(f"Timed out waiting for {description}: {last_error}")


class DemoClient:
    def __init__(self, url):
        self.http = httpx.Client(base_url=url, timeout=30, headers={"X-Admin-Token": TOKEN})

    def request(self, method, path, **kwargs):
        response = self.http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def status(self):
        return self.request("GET", "/admin/status")

    def caught_up(self, name=None):
        status = self.status()
        return next(
            (
                row
                for row in status["generations"]
                if row["status"] == "active"
                and (name is None or row["name"] == name)
                and row["cursor"] == status["catalog"]["sequence"]
                and not row["error"]
            ),
            None,
        )

    def search(self, query, mode="hybrid", **filters):
        return self.request(
            "POST", "/search", json={"query": query, "mode": mode, "limit": 10, **filters}
        )

    def put(self, sku, product, version=0, key=None):
        return self.request(
            "PUT",
            f"/admin/products/{sku}",
            json=product,
            headers={"If-Match": str(version), "Idempotency-Key": key or uuid4().hex},
        )

    def delete(self, sku, version):
        return self.request(
            "DELETE",
            f"/admin/products/{sku}",
            headers={"If-Match": str(version), "Idempotency-Key": uuid4().hex},
        )

    def clean(self, skus):
        for sku in skus:
            response = self.http.get(f"/products/{sku}")
            if response.status_code == 200:
                self.delete(sku, response.json()["version"])
            elif response.status_code != 404:
                response.raise_for_status()


def sample_product(**changes):
    return {
        "title": "Recovery keyboard",
        "description": "A quiet keyboard used to check reliable catalog updates.",
        "category": "test",
        "brand": "demo",
        "family": "recovery",
        "price_cents": 7500,
        "in_stock": True,
        **changes,
    }
