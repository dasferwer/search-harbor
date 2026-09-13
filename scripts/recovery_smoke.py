"""Остановить компоненты своего Compose-проекта и проверить восстановление."""

import json
import os
import subprocess
import threading
from pathlib import Path
from uuid import uuid4

from demo_client import DemoClient, sample_product, wait_for

ROOT = Path(__file__).resolve().parents[1]


def compose(*args, delay="0"):
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        env={**os.environ, "WORKER_AFTER_BULK_DELAY": delay},
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    ).stdout


def indexed(name, sku):
    # Читаем документ через realtime GET: до refresh он ещё может быть не виден в поиске.
    code = (
        "import json,urllib.request; print(urllib.request.urlopen('http://opensearch:9200/"
        + name
        + "/_doc/"
        + sku
        + "').read().decode())"
    )
    try:
        return json.loads(compose("exec", "-T", "api", "python", "-c", code)).get("found", False)
    except subprocess.CalledProcessError:
        return False


def main():
    client = DemoClient(os.getenv("BASE_URL", "http://localhost:8220"))
    prefix = "recovery-" + uuid4().hex
    skus = [prefix + "-offline", prefix + "-crash"]
    stop_queries, failures, observations = threading.Event(), [], []
    thread = None
    report = {}
    try:
        wait_for(client.caught_up, timeout=300)
        compose("stop", "--timeout", "20", "opensearch")
        offline = client.put(skus[0], sample_product())
        assert client.http.post("/search", json={"query": "keyboard"}).status_code == 503
        assert client.http.get("/ready").status_code == 503
        assert client.request("GET", f"/products/{skus[0]}")["version"] == offline["version"]
        compose("up", "-d", "--wait", "--wait-timeout", "120", "opensearch")
        active = wait_for(client.caught_up, description="index after OpenSearch outage")
        report["opensearch_outage"] = (
            "catalog write persisted; search returned 503; cursor recovered"
        )
        print("OpenSearch outage: passed", flush=True)

        compose("up", "-d", "--no-deps", "--force-recreate", "worker", delay="30")
        crashed = client.put(skus[1], sample_product(title="Crash recovery keyboard"))
        wait_for(
            lambda: indexed(active["name"], skus[1]),
            description="bulk acknowledged before cursor commit",
        )
        before = next(
            row for row in client.status()["generations"] if row["name"] == active["name"]
        )
        assert before["cursor"] < crashed["version"], (
            "Worker committed before the intended crash window"
        )
        compose("kill", "-s", "SIGKILL", "worker")
        compose("up", "-d", "--no-deps", "--force-recreate", "worker")
        wait_for(client.caught_up, description="replayed batch after worker crash")
        assert indexed(active["name"], skus[1])
        report["worker_crash"] = {
            "cursor_before_kill": before["cursor"],
            "event_replayed": crashed["version"],
            "result": "passed",
        }
        print("Worker crash after bulk: passed", flush=True)

        def query_during_rebuild():
            observer = DemoClient(client.http.base_url)
            try:
                while not stop_queries.is_set():
                    try:
                        response = observer.search("keyboard", "lexical")
                        if not response["items"]:
                            failures.append("empty catalog search")
                        observations.append(response["index"])
                    except Exception as error:
                        failures.append(str(error))
                    stop_queries.wait(0.1)
            finally:
                observer.http.close()

        thread = threading.Thread(target=query_during_rebuild, daemon=True)
        thread.start()
        rebuild = client.request("POST", "/admin/rebuilds")
        updated = client.put(skus[0], sample_product(price_cents=19000), offline["version"])
        client.delete(skus[1], crashed["version"])
        wait_for(
            lambda: client.caught_up(rebuild["name"]), description="concurrent rebuild", timeout=300
        )
        wait_for(lambda: rebuild["name"] in observations, description="query routed to new index")
        stop_queries.set()
        thread.join(timeout=35)
        assert not thread.is_alive() and not failures, failures
        assert active["name"] in observations and rebuild["name"] in observations
        result = client.search("Recovery keyboard", "lexical", category="test")
        assert [item["product"]["sku"] for item in result["items"]] == [skus[0]]
        assert result["items"][0]["product"]["version"] == updated["version"]
        assert indexed(active["name"], skus[0]), "Old index must remain for in-flight queries"
        report["rebuild"] = {
            "successful_searches": len(observations),
            "errors": len(failures),
            "updated_product_preserved": True,
            "deleted_product_absent": True,
        }
        print("Rebuild with concurrent search and catalog changes: passed", flush=True)
    finally:
        stop_queries.set()
        if thread:
            thread.join(timeout=35)
        # Проверка может оборваться на assert. Возвращаем стенд в обычный режим и сохраняем все его данные.
        compose("up", "-d", "--wait", "--wait-timeout", "120", "opensearch")
        compose("up", "-d", "--no-deps", "--force-recreate", "worker")
        client.clean(skus)
        wait_for(client.caught_up, description="cleanup events")
        client.http.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
