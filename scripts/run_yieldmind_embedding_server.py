#!/usr/bin/env python3
"""Serve a pinned sentence-transformer model on loopback for YieldMind."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import resource
import secrets
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


QWEN3_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN3_DIMENSIONS = 1024


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return int(peak if os.uname().sysname == "Darwin" else peak * 1024)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class EmbeddingRuntime:
    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        revision: str,
        dimensions: int,
        load_seconds: float,
    ) -> None:
        self.model = model
        self.model_id = model_id
        self.revision = revision
        self.dimensions = dimensions
        self.load_seconds = load_seconds
        self.started_at = time.time()
        self.encode_calls = 0
        self.encoded_texts = 0
        self.lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "model_id": self.model_id,
            "revision": self.revision,
            "dimensions": self.dimensions,
            "device": str(self.model.device),
            "load_seconds": round(self.load_seconds, 4),
            "versions": {
                package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "sentence-transformers")
            },
            "pid": os.getpid(),
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "encode_calls": self.encode_calls,
            "encoded_texts": self.encoded_texts,
            "uptime_seconds": round(time.time() - self.started_at, 3),
        }

    def encode(self, texts: list[str]) -> tuple[list[list[float]], float]:
        started = time.perf_counter()
        with self.lock:
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            self.encode_calls += 1
            self.encoded_texts += len(texts)
        if len(vectors.shape) != 2 or int(vectors.shape[1]) != self.dimensions:
            raise ValueError(f"Model returned shape {vectors.shape}; expected (*, {self.dimensions}).")
        return vectors.tolist(), time.perf_counter() - started


def handler_factory(runtime: EmbeddingRuntime, token: str, max_batch_size: int):
    class Handler(BaseHTTPRequestHandler):
        server_version = "YieldMindEmbedding/1.0"

        def _authorized(self) -> bool:
            return not token or secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}")

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if not self._authorized():
                self._send(401, {"ok": False, "error": "unauthorized"})
                return
            if self.path != "/health":
                self._send(404, {"ok": False, "error": "not found"})
                return
            self._send(200, runtime.health())

        def do_POST(self) -> None:
            if not self._authorized():
                self._send(401, {"ok": False, "error": "unauthorized"})
                return
            if self.path != "/embed":
                self._send(404, {"ok": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 8 * 1024 * 1024:
                    raise ValueError("request body must be between 1 byte and 8 MiB")
                payload = json.loads(self.rfile.read(length))
                texts = payload.get("texts")
                if not isinstance(texts, list) or not texts or len(texts) > max_batch_size:
                    raise ValueError(f"texts must be a non-empty list with at most {max_batch_size} items")
                if any(not isinstance(text, str) or not text.strip() or len(text) > 200_000 for text in texts):
                    raise ValueError("each text must be a non-empty string of at most 200000 characters")
                vectors, duration = runtime.encode(texts)
                self._send(
                    200,
                    {
                        "ok": True,
                        "kind": str(payload.get("kind") or "unspecified"),
                        "dimensions": runtime.dimensions,
                        "count": len(vectors),
                        "duration_seconds": round(duration, 6),
                        "vectors": vectors,
                    },
                )
            except Exception as exc:
                self._send(400, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model-id", default=QWEN3_MODEL_ID)
    parser.add_argument("--revision", default=QWEN3_REVISION)
    parser.add_argument("--dimensions", type=int, default=QWEN3_DIMENSIONS)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--hf-home", default="")
    parser.add_argument("--hf-endpoint", default="")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--token-env", default="YIELDMIND_EMBEDDING_TOKEN")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("Embedding server must bind to a loopback address.")
    if not args.revision or args.revision.lower() == "main":
        raise ValueError("An immutable model revision is required.")
    if args.max_batch_size < 1 or args.max_batch_size > 64:
        raise ValueError("max-batch-size must be between 1 and 64.")
    if args.hf_home:
        os.environ["HF_HOME"] = str(os.path.abspath(os.path.expanduser(args.hf_home)))
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from sentence_transformers import SentenceTransformer

    load_started = time.perf_counter()
    model = SentenceTransformer(
        args.model_id,
        revision=args.revision,
        local_files_only=not args.allow_model_download,
        tokenizer_kwargs={"padding_side": "left"},
        device="cpu",
    )
    actual_dimensions = int(model.get_sentence_embedding_dimension())
    if actual_dimensions != args.dimensions:
        raise ValueError(f"Model dimension mismatch: expected {args.dimensions}, got {actual_dimensions}.")
    load_seconds = time.perf_counter() - load_started
    runtime = EmbeddingRuntime(
        model,
        model_id=args.model_id,
        revision=args.revision,
        dimensions=args.dimensions,
        load_seconds=load_seconds,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        handler_factory(runtime, os.environ.get(args.token_env, ""), args.max_batch_size),
    )
    print(
        json.dumps(
            {
                "status": "ready",
                **runtime.health(),
                "host": args.host,
                "port": args.port,
                "model_download_allowed": bool(args.allow_model_download),
                "max_batch_size": args.max_batch_size,
                "hf_home": os.environ.get("HF_HOME", ""),
                "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    def stop_server(*_: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop_server)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
