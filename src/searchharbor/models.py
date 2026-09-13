import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from huggingface_hub import snapshot_download

from .config import MODEL_NAME, settings

SOURCES = {
    "embedding": {
        "repo": "qdrant/bge-small-en-v1.5-onnx-q",
        "revision": "52398278842ec682c6f32300af41344b1c0b0bb2",
        "files": [
            "config.json",
            "model_optimized.onnx",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ],
    },
    "reranker": {
        "repo": "Xenova/ms-marco-MiniLM-L-6-v2",
        "revision": "a09144355adeed5f58c8ed011d209bf8ee5a1fec",
        "files": [
            "config.json",
            "onnx/model.onnx",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ],
    },
}


def download():
    for name, source in SOURCES.items():
        snapshot_download(
            repo_id=source["repo"],
            revision=source["revision"],
            allow_patterns=source["files"],
            local_dir=str(Path(settings.model_cache) / name),
            token=False,
        )


def model_fingerprint(name):
    hashes = {}
    for filename in SOURCES[name]["files"]:
        path = Path(settings.model_cache) / name / filename
        with path.open("rb") as stream:
            hashes[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


class Models:
    def __init__(self):
        self.lock = threading.Lock()
        self.identity = model_fingerprint("embedding")
        self.embedding = TextEmbedding(
            MODEL_NAME,
            threads=2,
            providers=["CPUExecutionProvider"],
            specific_model_path=str(Path(settings.model_cache) / "embedding"),
        )
        self.reranker = None
        self.query = lru_cache(maxsize=512)(self._query)

    def documents(self, texts):
        with self.lock:
            return [vector.tolist() for vector in self.embedding.embed(texts, batch_size=32)]

    def _query(self, text):
        with self.lock:
            return next(self.embedding.query_embed(text)).tolist()

    def rerank(self, query, texts):
        with self.lock:
            if self.reranker is None:
                self.reranker = TextCrossEncoder(
                    "Xenova/ms-marco-MiniLM-L-6-v2",
                    threads=2,
                    providers=["CPUExecutionProvider"],
                    specific_model_path=str(Path(settings.model_cache) / "reranker"),
                )
            return [float(score) for score in self.reranker.rerank(query, texts, batch_size=16)]


def document_text(body):
    return f"{body['title']}. {body['description']} Category: {body['category']}."


if __name__ == "__main__":
    download()
    print(
        json.dumps(
            {
                name: {**source, "sha256": model_fingerprint(name)}
                for name, source in SOURCES.items()
            },
            indent=2,
        )
    )
