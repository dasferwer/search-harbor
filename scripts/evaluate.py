"""Сравнить фиксированные режимы поиска на небольшом авторском наборе запросов."""

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODES = ("lexical", "semantic", "hybrid", "rerank")
K = 5


def metrics(families: list[str], relevance: dict[str, int], k: int = K) -> dict[str, float]:
    relevant = {family for family, grade in relevance.items() if grade > 0}
    if not relevant or k < 1:
        raise ValueError("At least one relevant family and a positive cutoff are required")
    seen, gains, first = set(), [], 0
    for rank, family in enumerate(families[:k], 1):
        # Повтор одного семейства не должен улучшать оценку за счёт похожих вариантов.
        grade = relevance.get(family, 0) if family not in seen else 0
        gains.append((2**grade - 1) / math.log2(rank + 1))
        if grade > 0 and not first:
            first = rank
        seen.add(family)
    ideal = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(sorted(relevance.values(), reverse=True)[:k], 1)
    )
    return {
        "recall_at_5": len(seen & relevant) / len(relevant),
        "ndcg_at_5": sum(gains) / ideal,
        "mrr": 1 / first if first else 0.0,
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_queries(dataset: dict, catalog: list[dict]) -> None:
    families = {product["family"] for product in catalog}
    texts, ids = set(), set()
    for split, size in (("dev", 12), ("test", 24)):
        if len(dataset[split]) != size:
            raise ValueError(f"Frozen {split} split must contain {size} queries")
        for row in dataset[split]:
            text = " ".join(row["query"].casefold().split())
            if not text or text in texts or row["id"] in ids:
                raise ValueError("Query texts and IDs must be unique across both splits")
            texts.add(text)
            ids.add(row["id"])
            if row["kind"] not in {"lexical", "semantic", "typo"}:
                raise ValueError("Unknown query kind")
            labels = row["relevance"]
            if not labels or not set(labels) <= families:
                raise ValueError("Relevance labels must name known catalog families")
            if any(type(grade) is not int or not 1 <= grade <= 3 for grade in labels.values()):
                raise ValueError("Explicit relevance labels must be integer grades from 1 to 3")


def select_mode(dev_summary: dict[str, dict]) -> str:
    # Порядок разрешения ничьей задан заранее. Test не участвует в выборе режима.
    return max(
        MODES,
        key=lambda mode: (
            dev_summary[mode]["ndcg_at_5"],
            dev_summary[mode]["mrr"],
            dev_summary[mode]["recall_at_5"],
            -MODES.index(mode),
        ),
    )


class Benchmark:
    def __init__(self, client: httpx.Client, deadline: float, timeout: float):
        self.client = client
        self.deadline = deadline
        self.timeout = timeout
        self.snapshot = None

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Evaluation exceeded its total time budget")
        return remaining

    def request_timeout(self) -> httpx.Timeout:
        limit = min(self.timeout, self.remaining())
        return httpx.Timeout(limit, connect=min(5, limit))

    def wait_ready(self, wait_seconds: float) -> dict:
        stop = min(self.deadline, time.monotonic() + wait_seconds)
        last_error = "No response"
        while time.monotonic() < stop:
            try:
                response = self.client.get("/ready", timeout=self.request_timeout())
                if response.status_code == 200:
                    return response.json()
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = type(exc).__name__
            time.sleep(max(0, min(2, stop - time.monotonic())))
        raise TimeoutError(f"API did not become ready: {last_error}")

    def search(self, query: str, mode: str) -> tuple[dict, float]:
        start = time.perf_counter()
        response = self.client.post(
            "/search",
            json={"query": query, "mode": mode, "limit": K, "distinct_families": True},
            timeout=self.request_timeout(),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        result = response.json()
        snapshot = {key: result[key] for key in ("index", "indexed_sequence", "catalog_sequence")}
        if (
            result["lag_events"] != 0
            or snapshot["indexed_sequence"] != snapshot["catalog_sequence"]
        ):
            raise RuntimeError("Catalog changes have not reached the index; wait and rerun")
        if self.snapshot is None:
            self.snapshot = snapshot
        elif snapshot != self.snapshot:
            raise RuntimeError(
                "Catalog or active index changed during evaluation; rerun on a stable catalog"
            )
        families = [item["product"]["family"] for item in result["items"]]
        if len(families) > K or len(families) != len(set(families)):
            raise RuntimeError("Search violated the requested limit or family diversity")
        return result, elapsed_ms

    def run_split(self, queries: list[dict], repeats: int) -> tuple[dict, list[dict]]:
        measurements = {(row["id"], mode): [] for row in queries for mode in MODES}
        responses = {}
        for repeat in range(repeats):
            for position, row in enumerate(queries):
                offset = (position + repeat) % len(MODES)
                # Режимы идут по очереди, чтобы прогрев машины не совпадал с одним режимом.
                for mode in MODES[offset:] + MODES[:offset]:
                    result, elapsed_ms = self.search(row["query"], mode)
                    key = (row["id"], mode)
                    ranking = [item["product"]["family"] for item in result["items"]]
                    if key in responses:
                        previous = [item["product"]["family"] for item in responses[key]["items"]]
                        if ranking != previous:
                            raise RuntimeError(f"Unstable family ranking for {row['id']} / {mode}")
                    responses[key] = result
                    measurements[key].append(elapsed_ms)
        records = []
        for row in queries:
            for mode in MODES:
                key = (row["id"], mode)
                items = responses[key]["items"]
                records.append(
                    {
                        **row,
                        "mode": mode,
                        "returned": [
                            {"family": item["product"]["family"], "score": item["score"]}
                            for item in items
                        ],
                        "metrics": metrics(
                            [item["product"]["family"] for item in items], row["relevance"]
                        ),
                        "latency_ms": measurements[key],
                    }
                )
        summary = {}
        for mode in MODES:
            selected = [row for row in records if row["mode"] == mode]
            latencies = [latency for row in selected for latency in row["latency_ms"]]
            summary[mode] = {
                **{
                    key: statistics.mean(row["metrics"][key] for row in selected)
                    for key in ("recall_at_5", "ndcg_at_5", "mrr")
                },
                "p50_ms": percentile(latencies, 0.5),
                "p95_ms": percentile(latencies, 0.95),
                "query_count": len(selected),
                "request_count": len(latencies),
            }
        return summary, records


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8220")
    parser.add_argument("--queries", type=Path, default=ROOT / "data/queries.json")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/evaluation.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--ready-timeout", type=float, default=300)
    parser.add_argument("--max-duration", type=float, default=1800)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be between 1 and 20")
    if min(args.request_timeout, args.ready_timeout, args.max_duration) <= 0:
        parser.error("Timeouts must be positive")
    dataset = json.loads(args.queries.read_text())
    catalog = json.loads(args.catalog.read_text())
    validate_queries(dataset, catalog)
    started = datetime.now(UTC)
    with httpx.Client(base_url=args.url) as client:
        benchmark = Benchmark(client, time.monotonic() + args.max_duration, args.request_timeout)
        ready = benchmark.wait_ready(args.ready_timeout)
        for mode in MODES:
            for row in dataset["dev"][:2]:
                benchmark.search(row["query"], mode)
        dev_summary, dev_records = benchmark.run_split(dataset["dev"], args.repeats)
        recommended = select_mode(dev_summary)
        print(f"Dev completed. Selected before test: {recommended}", flush=True)
        test_summary, test_records = benchmark.run_split(dataset["test"], args.repeats)
    result = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "base_url": args.url,
        "benchmark": "Authored synthetic English catalog; not an independent human-labelled benchmark",
        "checksums": {
            "catalog_source_sha256": checksum(args.catalog),
            "queries_sha256": checksum(args.queries),
            "seed_code_sha256": checksum(ROOT / "src/searchharbor/seed.py"),
            "evaluation_code_sha256": checksum(Path(__file__)),
        },
        "catalog_source_family_count": len(catalog),
        "corpus_note": "Catalog checksum covers source family definitions, not a live database export. Seed creates near-identical variants; all relevance is measured per family.",
        "ready": ready,
        "stable_snapshot": benchmark.snapshot,
        "protocol": {
            "cutoff": K,
            "distinct_families": True,
            "repeats": args.repeats,
            "warmup_requests_per_mode": 2,
            "warmup_included": False,
            "concurrency": 1,
            "selection_order": [
                "dev NDCG@5",
                "dev MRR",
                "dev Recall@5",
                "fixed lexical/semantic/hybrid/rerank order",
            ],
            "mrr_cutoff": K,
            "latency": "Sequential warmed HTTP wall time including server and transport, linear interpolated percentiles; query repetitions may benefit from caches",
        },
        "recommended_mode_from_dev": recommended,
        "test_metrics_for_dev_selected_mode": test_summary[recommended],
        "splits": {
            "dev": {
                "kinds": dict(Counter(row["kind"] for row in dataset["dev"])),
                "summary": dev_summary,
                "queries": dev_records,
            },
            "test": {
                "kinds": dict(Counter(row["kind"] for row in dataset["test"])),
                "summary": test_summary,
                "queries": test_records,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recommended_mode_from_dev": recommended,
                "test": test_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
