"""Versioned domain knowledge ingestion and hybrid retrieval.

Chroma stores vectors while the configured YieldMind database stores auditable
document/chunk metadata. The default embedding remains deterministic and
offline; sentence-transformer models require an explicit profile and opt-in to
model downloads.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, model_validator

from yieldmind.database import PROJECT_ROOT, YieldMindStore, connect, json_dumps, json_loads


DEFAULT_INDEX_VERSION = "yieldmind-chroma-hashing-v1"
DEFAULT_COLLECTION = "yieldmind_domain_v1"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "chroma"
SPLIT_VERSION = "recursive_chars_1200_180_v1"
SUPPORTED_SUFFIXES = {".md", ".txt"}
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
LATIN_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


class EmbeddingProfile(BaseModel):
    """Immutable settings that define one compatible vector index."""

    provider: Literal["local_hashing", "sentence_transformers", "http_sentence_transformers"] = "local_hashing"
    model_id: str = "local_hashing_v1"
    revision: str = "builtin-v1"
    dimensions: int = Field(default=384, ge=32, le=8192)
    normalize: bool = True
    metric: Literal["cosine"] = "cosine"
    query_instruction: str = ""
    document_instruction: str = ""
    index_version: str = DEFAULT_INDEX_VERSION

    @model_validator(mode="after")
    def validate_reproducible_model_profile(self) -> "EmbeddingProfile":
        if self.provider in {"sentence_transformers", "http_sentence_transformers"}:
            if self.model_id == "local_hashing_v1":
                raise ValueError("sentence_transformers profile requires an explicit model_id.")
            if not self.revision or self.revision.lower() == "main":
                raise ValueError(
                    "sentence_transformers profile requires an immutable model revision, not the floating 'main' branch."
                )
        return self

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        return _sha256_text(json_dumps(payload))[:12]


DEFAULT_EMBEDDING_PROFILE = EmbeddingProfile()


class KnowledgeIngestRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    index_version: str = DEFAULT_INDEX_VERSION
    chunk_size: int = Field(default=1200, ge=200, le=4000)
    chunk_overlap: int = Field(default=180, ge=0, le=1000)


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    index_version: str = DEFAULT_INDEX_VERSION
    document_id: str | None = None
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "vector"


class KnowledgeDeleteRequest(BaseModel):
    document_id: str
    index_version: str = DEFAULT_INDEX_VERSION


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_id(*parts: Any, prefix: str = "") -> str:
    digest = _sha256_text("::".join(str(part) for part in parts))[:24]
    return f"{prefix}{digest}" if prefix else digest


def _chroma_where(index_version: str, document_id: str | None = None) -> dict[str, Any]:
    clauses = [{"index_version": index_version}]
    if document_id:
        clauses.append({"document_id": document_id})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _read_text_file(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type {suffix!r}; supported: {sorted(SUPPORTED_SUFFIXES)}")
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(encoding), _sha256_bytes(data)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode document with utf-8/gb18030/gbk: {path}")


class EmbeddingFunction(Protocol):
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbeddingFunction:
    """Small deterministic embedding for offline RAG tests.

    This is intentionally not advertised as Qwen/BGE quality. It gives Chroma a
    reproducible embedding path without downloading external model weights.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = int(dimensions)
        self.document_encode_calls = 0
        self.query_encode_calls = 0

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_encode_calls += 1
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_encode_calls += 1
        return self._embed_one(text)


class SentenceTransformerEmbeddingFunction:
    """Lazy sentence-transformers adapter with explicit download control."""

    def __init__(self, profile: EmbeddingProfile, *, allow_model_download: bool = False) -> None:
        if profile.provider != "sentence_transformers":
            raise ValueError("SentenceTransformerEmbeddingFunction requires provider='sentence_transformers'.")
        self.profile = profile
        self.dimensions = profile.dimensions
        self.allow_model_download = allow_model_download
        self._model: Any = None
        self.document_encode_calls = 0
        self.query_encode_calls = 0

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {"local_files_only": not self.allow_model_download}
            if self.profile.revision and self.profile.revision != "main":
                kwargs["revision"] = self.profile.revision
            try:
                self._model = SentenceTransformer(self.profile.model_id, **kwargs)
            except TypeError as exc:
                if not self.allow_model_download:
                    raise RuntimeError(
                        "Installed sentence-transformers cannot enforce local_files_only; "
                        "upgrade it in an isolated embedding environment before loading this model."
                    ) from exc
                kwargs.pop("local_files_only", None)
                self._model = SentenceTransformer(self.profile.model_id, **kwargs)
            actual_dimension = int(self._model.get_sentence_embedding_dimension())
            if actual_dimension != self.profile.dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch: profile={self.profile.dimensions}, model={actual_dimension}."
                )
        return self._model

    def _encode(self, texts: list[str], *, instruction: str) -> list[list[float]]:
        prepared = [f"{instruction}{text}" if instruction else text for text in texts]
        vectors = self._load().encode(
            prepared,
            normalize_embeddings=self.profile.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_encode_calls += 1
        return self._encode(texts, instruction=self.profile.document_instruction)

    def embed_query(self, text: str) -> list[float]:
        self.query_encode_calls += 1
        return self._encode([text], instruction=self.profile.query_instruction)[0]


class HttpEmbeddingFunction:
    """Client for an isolated loopback embedding process."""

    def __init__(
        self,
        profile: EmbeddingProfile,
        *,
        endpoint: str,
        token: str = "",
        timeout_seconds: float = 300.0,
    ) -> None:
        if profile.provider != "http_sentence_transformers":
            raise ValueError("HttpEmbeddingFunction requires provider='http_sentence_transformers'.")
        self.profile = profile
        self.dimensions = profile.dimensions
        self.endpoint = endpoint.rstrip("/")
        parsed_endpoint = urllib.parse.urlsplit(self.endpoint)
        if parsed_endpoint.scheme != "http" or parsed_endpoint.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("Embedding endpoint must use HTTP on a loopback host.")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.document_encode_calls = 0
        self.query_encode_calls = 0
        health = self._request("GET", "/health")
        expected = {
            "model_id": profile.model_id,
            "revision": profile.revision,
            "dimensions": profile.dimensions,
        }
        actual = {key: health.get(key) for key in expected}
        if actual != expected or not health.get("ok"):
            raise ValueError(f"Embedding service identity mismatch: expected={expected}, actual={actual}.")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json_dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(f"{self.endpoint}{path}", data=data, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                return json_loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embedding service HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding service unavailable at {self.endpoint}: {exc.reason}") from exc

    def _encode(self, texts: list[str], *, instruction: str, kind: str) -> list[list[float]]:
        prepared = [f"{instruction}{text}" if instruction else text for text in texts]
        response = self._request("POST", "/embed", {"texts": prepared, "kind": kind})
        vectors = response.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("Embedding service returned an invalid vector batch.")
        if any(not isinstance(vector, list) or len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError(f"Embedding service returned a vector with dimension other than {self.dimensions}.")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_encode_calls += 1
        return self._encode(texts, instruction=self.profile.document_instruction, kind="documents")

    def embed_query(self, text: str) -> list[float]:
        self.query_encode_calls += 1
        return self._encode([text], instruction=self.profile.query_instruction, kind="query")[0]


def build_embedding(
    profile: EmbeddingProfile,
    *,
    allow_model_download: bool = False,
) -> EmbeddingFunction:
    if profile.provider == "local_hashing":
        return HashingEmbeddingFunction(profile.dimensions)
    if profile.provider == "sentence_transformers":
        return SentenceTransformerEmbeddingFunction(profile, allow_model_download=allow_model_download)
    raise ValueError("http_sentence_transformers profiles require an injected HttpEmbeddingFunction.")


@dataclass
class IngestedDocument:
    document_id: str
    title: str
    source_path: str
    document_hash: str
    document_version: str
    index_version: str
    chunk_count: int
    status: str


class KnowledgeBase:
    def __init__(
        self,
        *,
        store: YieldMindStore | None = None,
        chroma_dir: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        embedding: EmbeddingFunction | None = None,
        embedding_profile: EmbeddingProfile | None = None,
        allow_model_download: bool = False,
    ) -> None:
        self.store = store or YieldMindStore()
        self.chroma_dir = Path(chroma_dir or DEFAULT_CHROMA_DIR)
        self.embedding_profile = embedding_profile or DEFAULT_EMBEDDING_PROFILE
        self.embedding = embedding or build_embedding(
            self.embedding_profile,
            allow_model_download=allow_model_download,
        )
        if int(self.embedding.dimensions) != self.embedding_profile.dimensions:
            raise ValueError(
                "Embedding implementation dimension does not match the embedding profile: "
                f"{self.embedding.dimensions} != {self.embedding_profile.dimensions}."
            )
        base_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name).strip("_-") or "yieldmind"
        # A Chroma collection cannot contain vectors with different dimensions.
        self.collection_name = f"{base_name[:48]}_{self.embedding_profile.fingerprint()}"

    def _validate_index_version(self, index_version: str) -> None:
        if index_version != self.embedding_profile.index_version:
            raise ValueError(
                "Request index_version does not match the active embedding profile: "
                f"{index_version!r} != {self.embedding_profile.index_version!r}."
            )

    def _collection(self):
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        os.environ.setdefault("CHROMA_TELEMETRY", "False")
        import chromadb
        from chromadb.config import Settings

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": self.embedding_profile.metric,
                "embedding_provider": self.embedding_profile.provider,
                "embedding_model": self.embedding_profile.model_id,
                "embedding_revision": self.embedding_profile.revision,
                "embedding_dimensions": self.embedding_profile.dimensions,
                "embedding_profile": self.embedding_profile.fingerprint(),
                "index_version": self.embedding_profile.index_version,
            },
        )

    def _split(self, text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n# ", "\n\n", "\n", "。", ". ", " ", ""],
        )
        return [chunk for chunk in splitter.split_text(text) if chunk.strip()]

    def ingest(self, request: KnowledgeIngestRequest) -> dict[str, Any]:
        self._validate_index_version(request.index_version)
        collection = self._collection()
        results: list[dict[str, Any]] = []
        for raw_path in request.paths:
            path = Path(raw_path).expanduser().resolve()
            started = time.time()
            if not path.exists():
                results.append({"source_path": str(path), "status": "failed", "error": "path does not exist"})
                continue
            try:
                text, document_hash = _read_text_file(path)
                title = path.stem
                document_version = document_hash[:16]
                document_id = _stable_id(str(path), document_hash, request.index_version, prefix="doc_")
                existing = self.get_document(document_id)
                indexed = False
                if existing and existing.get("status") == "available":
                    stored = collection.get(
                        where=_chroma_where(request.index_version, document_id),
                        limit=1,
                    )
                    indexed = bool(stored.get("ids"))
                if existing and existing.get("status") == "available" and indexed:
                    results.append({**existing, "status": "available", "idempotent": True})
                    continue

                chunks = self._split(text, chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap)
                now = time.time()
                ids: list[str] = []
                documents: list[str] = []
                metadatas: list[dict[str, Any]] = []
                db_rows: list[tuple[Any, ...]] = []
                for idx, chunk in enumerate(chunks):
                    text_hash = _sha256_text(chunk)
                    chunk_id = _stable_id(document_id, document_version, SPLIT_VERSION, idx, text_hash, prefix="chk_")
                    ids.append(chunk_id)
                    documents.append(chunk)
                    metadata = {
                        "document_id": document_id,
                        "document_version": document_version,
                        "chunk_id": chunk_id,
                        "chunk_index": idx,
                        "title": title,
                        "source_path": str(path),
                        "text_hash": text_hash,
                        "index_version": request.index_version,
                        "split_version": SPLIT_VERSION,
                    }
                    metadatas.append(metadata)
                    db_rows.append(
                        (
                            chunk_id,
                            document_id,
                            document_version,
                            idx,
                            title,
                            str(path),
                            "",
                            text_hash,
                            request.index_version,
                            chunk,
                            now,
                        )
                    )

                embeddings = self.embedding.embed_documents(documents)
                if ids:
                    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
                with connect(self.store.db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO yieldmind_documents
                            (document_id, title, source_path, source_type, document_hash,
                             document_version, status, index_version, chunk_count, error,
                             created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(document_id) DO UPDATE SET
                            title=excluded.title,
                            source_path=excluded.source_path,
                            source_type=excluded.source_type,
                            document_hash=excluded.document_hash,
                            document_version=excluded.document_version,
                            status=excluded.status,
                            index_version=excluded.index_version,
                            chunk_count=excluded.chunk_count,
                            error=excluded.error,
                            updated_at=excluded.updated_at
                        """,
                        (
                            document_id,
                            title,
                            str(path),
                            path.suffix.lower().lstrip("."),
                            document_hash,
                            document_version,
                            "available",
                            request.index_version,
                            len(chunks),
                            "",
                            now,
                            now,
                        ),
                    )
                    conn.execute("DELETE FROM yieldmind_document_chunks WHERE document_id=?", (document_id,))
                    conn.executemany(
                        """
                        INSERT INTO yieldmind_document_chunks
                            (chunk_id, document_id, document_version, chunk_index, title,
                             source_path, section, text_hash, index_version, text, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(chunk_id) DO UPDATE SET
                            document_id=excluded.document_id,
                            document_version=excluded.document_version,
                            chunk_index=excluded.chunk_index,
                            title=excluded.title,
                            source_path=excluded.source_path,
                            section=excluded.section,
                            text_hash=excluded.text_hash,
                            index_version=excluded.index_version,
                            text=excluded.text,
                            created_at=excluded.created_at
                        """,
                        db_rows,
                    )
                results.append(
                    IngestedDocument(
                        document_id=document_id,
                        title=title,
                        source_path=str(path),
                        document_hash=document_hash,
                        document_version=document_version,
                        index_version=request.index_version,
                        chunk_count=len(chunks),
                        status="available",
                    ).__dict__
                    | {"duration_seconds": round(time.time() - started, 4), "idempotent": False}
                )
            except Exception as exc:
                results.append({"source_path": str(path), "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return {
            "index_version": request.index_version,
            "collection": self.collection_name,
            "embedding_profile": self.embedding_profile.model_dump(mode="json"),
            "embedding_profile_fingerprint": self.embedding_profile.fingerprint(),
            "embedding_model": self.embedding_profile.model_id,
            "split_version": SPLIT_VERSION,
            "documents": results,
        }

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with connect(self.store.db_path) as conn:
            row = conn.execute("SELECT * FROM yieldmind_documents WHERE document_id=?", (document_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with connect(self.store.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM yieldmind_documents ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _lexical_tokens(text: str) -> list[str]:
        lowered = text.lower()
        tokens = LATIN_TOKEN_RE.findall(lowered)
        for run in CJK_RUN_RE.findall(lowered):
            tokens.extend(run)
            tokens.extend(run[index : index + 2] for index in range(max(0, len(run) - 1)))
        return tokens

    def _active_chunks(self, request: KnowledgeSearchRequest) -> list[dict[str, Any]]:
        sql = """
            SELECT c.* FROM yieldmind_document_chunks c
            JOIN yieldmind_documents d ON d.document_id=c.document_id
            WHERE c.index_version=? AND d.status='available'
        """
        params: list[Any] = [request.index_version]
        if request.document_id:
            sql += " AND c.document_id=?"
            params.append(request.document_id)
        sql += " ORDER BY c.document_id, c.chunk_index"
        with connect(self.store.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _hit_from_metadata(
        *,
        chunk_id: str,
        text: str,
        metadata: dict[str, Any],
        score: float | None,
        distance: float | None = None,
    ) -> dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "document_id": metadata.get("document_id"),
            "document_version": metadata.get("document_version"),
            "title": metadata.get("title"),
            "source_path": metadata.get("source_path"),
            "chunk_index": metadata.get("chunk_index"),
            "text_hash": metadata.get("text_hash"),
            "index_version": metadata.get("index_version"),
            "score": score,
            "distance": distance,
            "text": text,
        }

    def _vector_hits(self, request: KnowledgeSearchRequest, *, candidate_k: int) -> list[dict[str, Any]]:
        collection = self._collection()
        where = _chroma_where(request.index_version, request.document_id)
        query_embedding = self.embedding.embed_query(request.query)
        try:
            n_results = min(candidate_k, max(1, int(collection.count())))
        except Exception:
            n_results = candidate_k
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            score = 1.0 - float(distance) if distance is not None else None
            hits.append(
                self._hit_from_metadata(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=meta,
                    score=score,
                    distance=float(distance) if distance is not None else None,
                )
            )
        return hits

    def _bm25_hits(self, request: KnowledgeSearchRequest, *, candidate_k: int) -> list[dict[str, Any]]:
        from rank_bm25 import BM25Plus

        chunks = self._active_chunks(request)
        if not chunks:
            return []
        tokenized_corpus = [self._lexical_tokens(str(chunk["text"])) for chunk in chunks]
        query_tokens = self._lexical_tokens(request.query)
        if not query_tokens:
            return []
        scores = BM25Plus(tokenized_corpus).get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))
        hits: list[dict[str, Any]] = []
        for index, score in ranked:
            if float(score) <= 0:
                continue
            row = chunks[index]
            hits.append(
                self._hit_from_metadata(
                    chunk_id=str(row["chunk_id"]),
                    text=str(row["text"]),
                    metadata=row,
                    score=float(score),
                )
            )
            if len(hits) >= candidate_k:
                break
        return hits

    @staticmethod
    def _rrf(vector_hits: list[dict[str, Any]], bm25_hits: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}
        channels: dict[str, list[str]] = {}
        for channel, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
            for rank, hit in enumerate(hits, start=1):
                chunk_id = str(hit["chunk_id"])
                by_id.setdefault(chunk_id, hit.copy())
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
                channels.setdefault(chunk_id, []).append(channel)
        ranked_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
        result = []
        for chunk_id in ranked_ids:
            hit = by_id[chunk_id]
            hit["score"] = scores[chunk_id]
            hit["distance"] = None
            hit["retrieval_channels"] = channels[chunk_id]
            result.append(hit)
        return result

    def search(self, request: KnowledgeSearchRequest) -> dict[str, Any]:
        self._validate_index_version(request.index_version)
        started = time.perf_counter()
        candidate_k = min(60, max(request.top_k, request.top_k * 3))
        if request.retrieval_mode == "vector":
            hits = self._vector_hits(request, candidate_k=request.top_k)
        elif request.retrieval_mode == "bm25":
            hits = self._bm25_hits(request, candidate_k=request.top_k)
        else:
            vector_hits = self._vector_hits(request, candidate_k=candidate_k)
            bm25_hits = self._bm25_hits(request, candidate_k=candidate_k)
            hits = self._rrf(vector_hits, bm25_hits, top_k=request.top_k)
        return {
            "query": request.query,
            "top_k": request.top_k,
            "index_version": request.index_version,
            "retrieval_mode": request.retrieval_mode,
            "lexical_algorithm": "bm25_plus" if request.retrieval_mode in {"bm25", "hybrid"} else None,
            "embedding_profile": self.embedding_profile.model_dump(mode="json"),
            "embedding_profile_fingerprint": self.embedding_profile.fingerprint(),
            "embedding_model": self.embedding_profile.model_id,
            "hits": hits[: request.top_k],
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "note": (
                "Offline deterministic hashing vector baseline; no external embedding model call."
                if self.embedding_profile.provider == "local_hashing"
                else "Sentence-transformer retrieval using the explicitly configured embedding profile."
            ),
        }

    def delete_document(self, request: KnowledgeDeleteRequest) -> dict[str, Any]:
        self._validate_index_version(request.index_version)
        doc = self.get_document(request.document_id)
        if not doc:
            return {"ok": False, "error": f"document not found: {request.document_id}"}
        collection = self._collection()
        collection.delete(where=_chroma_where(request.index_version, request.document_id))
        now = time.time()
        with connect(self.store.db_path) as conn:
            conn.execute(
                "UPDATE yieldmind_documents SET status='deleted', updated_at=? WHERE document_id=?",
                (now, request.document_id),
            )
        return {"ok": True, "document_id": request.document_id, "status": "deleted"}


def evaluate_retrieval(
    kb: KnowledgeBase,
    cases: list[dict[str, Any]],
    *,
    top_k: int = 5,
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "vector",
) -> dict[str, Any]:
    rows = []
    reciprocal_ranks = []
    recalls = []
    citation_hits = []
    latencies = []
    for case in cases:
        expected_terms = [str(term).lower() for term in case.get("expected_terms", [])]
        expected_sources = {str(path) for path in case.get("expected_source_paths", [])}
        result = kb.search(
            KnowledgeSearchRequest(
                query=case["query"],
                top_k=top_k,
                index_version=kb.embedding_profile.index_version,
                retrieval_mode=retrieval_mode,
            )
        )
        rank = None
        for idx, hit in enumerate(result["hits"], start=1):
            text = str(hit.get("text") or "").lower()
            source_path = str(hit.get("source_path") or "")
            source_matches = not expected_sources or any(
                source_path == expected or source_path.endswith(f"/{expected}") for expected in expected_sources
            )
            terms_match = not expected_terms or all(term in text for term in expected_terms)
            if source_matches and terms_match:
                rank = idx
                break
        rows.append(
            {
                "id": case.get("id", ""),
                "query": case["query"],
                "expected_source_paths": sorted(expected_sources),
                "expected_terms": expected_terms,
                "rank": rank,
                "passed": rank is not None,
                "top_source_path": result["hits"][0].get("source_path") if result["hits"] else None,
                "latency_ms": result["latency_ms"],
            }
        )
        recalls.append(1.0 if rank is not None else 0.0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        citation_hits.append(1.0 if rank == 1 else 0.0)
        latencies.append(float(result["latency_ms"]))
    return {
        "case_count": len(cases),
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "citation_accuracy_at_1": sum(citation_hits) / len(citation_hits) if citation_hits else 0.0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "cases": rows,
    }
