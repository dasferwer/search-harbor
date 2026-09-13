import importlib.util
import json
import math
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


@pytest.fixture(autouse=True)
def clean_state():
    # Формулам и проверке отчёта не нужны ни база, ни очистка поискового индекса.
    yield


def test_perfect_and_empty_rankings():
    relevance = {"strong": 3, "related": 1}
    assert evaluation.metrics(["strong", "related"], relevance) == {
        "recall_at_5": 1,
        "ndcg_at_5": 1,
        "mrr": 1,
    }
    assert evaluation.metrics([], relevance) == {"recall_at_5": 0, "ndcg_at_5": 0, "mrr": 0}


def test_graded_relevance_rank_and_cutoff():
    relevance = {"strong": 3, "related": 1}
    result = evaluation.metrics(["other", "related", "strong"], relevance)
    assert result["recall_at_5"] == 1
    assert result["mrr"] == 0.5
    assert result["ndcg_at_5"] == pytest.approx((1 / math.log2(3) + 7 / 2) / (7 + 1 / math.log2(3)))
    assert evaluation.metrics(["other"] * 5 + ["strong"], relevance)["recall_at_5"] == 0


def test_duplicate_families_cannot_inflate_metrics():
    result = evaluation.metrics(["strong", "strong", "strong"], {"strong": 3, "related": 1})
    assert result["recall_at_5"] == 0.5
    assert result["ndcg_at_5"] == pytest.approx(7 / (7 + 1 / math.log2(3)))


def test_frozen_labels_have_no_query_overlap_and_only_known_families():
    dataset = json.loads((ROOT / "data/queries.json").read_text())
    catalog = json.loads((ROOT / "data/catalog.json").read_text())
    evaluation.validate_queries(dataset, catalog)
    dataset["test"][0]["query"] = dataset["dev"][0]["query"].upper()
    with pytest.raises(ValueError, match="unique"):
        evaluation.validate_queries(dataset, catalog)


def test_mode_selection_uses_dev_quality_and_fixed_tie_break():
    dev = {mode: {"ndcg_at_5": 0.5, "mrr": 1, "recall_at_5": 1} for mode in evaluation.MODES}
    assert evaluation.select_mode(dev) == "lexical"
    dev["semantic"]["ndcg_at_5"] = 0.6
    assert evaluation.select_mode(dev) == "semantic"


def test_percentile_uses_linear_interpolation():
    assert evaluation.percentile([40, 10, 30, 20], 0.5) == 25
    assert evaluation.percentile([40, 10, 30, 20], 0.95) == pytest.approx(38.5)


@pytest.mark.parametrize("change", [{"lag_events": 1}, {"index": "catalog-new"}])
def test_live_snapshot_changes_are_rejected(change):
    state = {
        "index": "catalog-old",
        "indexed_sequence": 1,
        "catalog_sequence": 1,
        "lag_events": 0,
        "items": [],
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=state))
    with httpx.Client(transport=transport, base_url="http://test") as client:
        benchmark = evaluation.Benchmark(client, time.monotonic() + 30, 5)
        benchmark.search("test", "lexical")
        state.update(change)
        with pytest.raises(RuntimeError):
            benchmark.search("test", "lexical")


def test_total_timeout_is_checked_before_search():
    with httpx.Client(base_url="http://test") as client:
        benchmark = evaluation.Benchmark(client, time.monotonic() - 1, 5)
        with pytest.raises(TimeoutError, match="total time budget"):
            benchmark.search("test", "lexical")
