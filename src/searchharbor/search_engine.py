import json

import httpx

from .config import MODEL_DIM, settings


class MissingIndex(RuntimeError):
    pass


class SearchEngine:
    def __init__(self, url=None):
        self.client = httpx.AsyncClient(base_url=url or settings.opensearch_url, timeout=15)

    async def close(self):
        await self.client.aclose()

    async def request(self, method, path, **kwargs):
        response = await self.client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    async def ensure_index(self, name, *, create=True):
        response = await self.client.head("/" + name)
        if response.status_code == 200:
            return
        if response.status_code != 404:
            response.raise_for_status()
        if not create:
            raise MissingIndex(f"Index {name} is missing; request a rebuild")
        response = await self.client.put(
            "/" + name,
            json={
                "settings": {
                    "index.knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "1s",
                },
                "mappings": {
                    "dynamic": "strict",
                    "properties": {
                        "sku": {"type": "keyword"},
                        "version": {"type": "long"},
                        "deleted": {"type": "boolean"},
                        "title": {"type": "text", "analyzer": "english"},
                        "description": {"type": "text", "analyzer": "english"},
                        "category": {"type": "keyword"},
                        "brand": {"type": "keyword"},
                        "family": {"type": "keyword"},
                        "price_cents": {"type": "integer"},
                        "in_stock": {"type": "boolean"},
                        "vector": {
                            "type": "knn_vector",
                            "dimension": MODEL_DIM,
                            "method": {
                                "name": "hnsw",
                                "engine": "lucene",
                                "space_type": "cosinesimil",
                            },
                        },
                    },
                },
            },
        )
        if (
            response.status_code == 400
            and response.json().get("error", {}).get("type") == "resource_already_exists_exception"
        ):
            return
        response.raise_for_status()

    async def bulk(self, name, documents):
        lines = []
        for document in documents:
            lines.extend(
                [
                    json.dumps(
                        {
                            "index": {
                                "_index": name,
                                "_id": document["sku"],
                                "version": document["version"],
                                "version_type": "external_gte",
                            }
                        }
                    ),
                    json.dumps(document),
                ]
            )
        result = await self.request(
            "POST",
            "/_bulk",
            content="\n".join(lines) + "\n",
            headers={"Content-Type": "application/x-ndjson"},
        )
        if len(result["items"]) != len(documents):
            raise RuntimeError("Bulk response has an unexpected number of items")
        failures = [
            item["index"]
            for item in result["items"]
            if item["index"]["status"] not in (200, 201)
            and not (
                item["index"]["status"] == 409
                and item["index"].get("error", {}).get("type")
                == "version_conflict_engine_exception"
            )
        ]
        if failures:
            raise RuntimeError(
                f"Bulk indexing rejected {len(failures)} documents: {failures[0].get('error', {}).get('type')}"
            )

    async def refresh(self, name):
        result = await self.request("POST", f"/{name}/_refresh")
        if result.get("_shards", {}).get("failed", 0):
            raise RuntimeError("Refresh did not complete on all shards")

    async def search(self, name, request, vector=None):
        filters = [
            {"term": {"deleted": False}},
            {"range": {"price_cents": {"gte": request.min_price, "lte": request.max_price}}},
        ]
        for field in ("category", "brand"):
            if value := getattr(request, field):
                filters.append({"term": {field: value}})
        if request.in_stock:
            filters.append({"term": {"in_stock": True}})
        if vector is None:
            query = {
                "bool": {
                    "filter": filters,
                    "must": [
                        {
                            "multi_match": {
                                "query": request.query,
                                "fields": ["title^3", "description"],
                                "fuzziness": "AUTO",
                                "operator": "or",
                            }
                        }
                    ],
                }
            }
        else:
            query = {
                "knn": {
                    "vector": {
                        "vector": vector,
                        "k": 1000 if request.distinct_families else 100,
                        "filter": {"bool": {"filter": filters}},
                    }
                }
            }
        body = {
            "query": query,
            "size": 100,
            "_source": {"excludes": ["vector"]},
            "sort": [{"_score": "desc"}, {"sku": "asc"}],
        }
        if request.distinct_families:
            body["collapse"] = {"field": "family"}
        result = await self.request("POST", f"/{name}/_search", json=body)
        if result.get("timed_out") or result.get("_shards", {}).get("failed", 0):
            raise RuntimeError("Search did not complete on all shards")
        return result["hits"]["hits"]
