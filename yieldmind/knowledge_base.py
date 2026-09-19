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
from yieldmind.document_loader import load_document


DEFAULT_INDEX_VERSION = "yieldmind-chroma-hashing-v1"
DEFAULT_COLLECTION = "yieldmind_domain_v1"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "chroma"
QWEN3_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN3_INDEX_VERSION = "yieldmind-qwen3-embedding-0.6b-97b0c614-1024-v1"
QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a query about yield-stress modeling and the YieldMind system, "
    "retrieve relevant passages that answer the query\nQuery:"
)
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
LATIN_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def split_version_for(chunk_size: int, chunk_overlap: int) -> str:
    return f"format_aware_recursive_chars_{int(chunk_size)}_{int(chunk_overlap)}_v3"


SPLIT_VERSION = split_version_for(1200, 180)


def split_knowledge_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n# ", "\n\n", "\n", "。", ". ", " ", ""],
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


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


def qwen3_embedding_profile(*, http: bool = True) -> EmbeddingProfile:
    return EmbeddingProfile(
        provider="http_sentence_transformers" if http else "sentence_transformers",
        model_id=QWEN3_MODEL_ID,
        revision=QWEN3_REVISION,
        dimensions=1024,
        normalize=True,
        metric="cosine",
        query_instruction=QWEN3_QUERY_INSTRUCTION,
        document_instruction="",
        index_version=QWEN3_INDEX_VERSION,
    )


class KnowledgeSourceMetadata(BaseModel):
    corpus: Literal["project", "literature"] = "project"
    title: str = ""
    source_url: str = ""
    doi: str = ""
    license: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_provenance(self) -> "KnowledgeSourceMetadata":
        self.title = self.title.strip()
        self.source_url = self.source_url.strip()
        self.doi = self.doi.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        self.license = self.license.strip()
        return self


class KnowledgeIngestRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    index_version: str = DEFAULT_INDEX_VERSION
    chunk_size: int = Field(default=1200, ge=200, le=4000)
    chunk_overlap: int = Field(default=180, ge=0, le=1000)
    source_metadata: dict[str, KnowledgeSourceMetadata] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "KnowledgeIngestRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    index_version: str = DEFAULT_INDEX_VERSION
    document_id: str | None = None
    corpus: Literal["project", "literature"] | None = None
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "vector"


class KnowledgeDeleteRequest(BaseModel):
    document_id: str
    index_version: str = DEFAULT_INDEX_VERSION


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_id(*parts: Any, prefix: str = "") -> str:
    digest = _sha256_text("::".join(str(part) for part in parts))[:24]
    return f"{prefix}{digest}" if prefix else digest


def _chroma_where(
    index_version: str,
    document_id: str | None = None,
    corpus: str | None = None,
) -> dict[str, Any]:
    clauses = [{"index_version": index_version}]
    if document_id:
        clauses.append({"document_id": document_id})
    if corpus:
        clauses.append({"corpus": corpus})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


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
        batch_size: int = 8,
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
        if batch_size < 1 or batch_size > 64:
            raise ValueError("Embedding batch_size must be between 1 and 64.")
        self.batch_size = int(batch_size)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.document_encode_calls = 0
        self.query_encode_calls = 0
        self.http_encode_calls = 0
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
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            response = self._request("POST", "/embed", {"texts": batch, "kind": kind})
            self.http_encode_calls += 1
            batch_vectors = response.get("vectors")
            if not isinstance(batch_vectors, list) or len(batch_vectors) != len(batch):
                raise RuntimeError("Embedding service returned an invalid vector batch.")
            if any(not isinstance(vector, list) or len(vector) != self.dimensions for vector in batch_vectors):
                raise RuntimeError(
                    f"Embedding service returned a vector with dimension other than {self.dimensions}."
                )
            vectors.extend(batch_vectors)
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
        return split_knowledge_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @staticmethod
    def _metadata_for_path(
        request: KnowledgeIngestRequest,
        raw_path: str,
        resolved_path: Path,
    ) -> KnowledgeSourceMetadata:
        for key in (raw_path, str(resolved_path), resolved_path.name):
            metadata = request.source_metadata.get(key)
            if metadata is not None:
                return metadata
        return KnowledgeSourceMetadata()

    def _document_by_source_path(self, source_path: str, index_version: str) -> dict[str, Any] | None:
        with connect(self.store.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM yieldmind_documents
                WHERE source_path=? AND index_version=?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (source_path, index_version),
            ).fetchone()
        return dict(row) if row else None

    def ingest(self, request: KnowledgeIngestRequest) -> dict[str, Any]:
        self._validate_index_version(request.index_version)
        collection = self._collection()
        split_version = split_version_for(request.chunk_size, request.chunk_overlap)
        results: list[dict[str, Any]] = []
        for raw_path in request.paths:
            path = Path(raw_path).expanduser().resolve()
            started = time.time()
            if not path.exists():
                results.append({"source_path": str(path), "status": "failed", "error": "path does not exist"})
                continue
            try:
                source_metadata = self._metadata_for_path(request, raw_path, path)
                loaded = load_document(path, title=source_metadata.title)
                document_hash = loaded.document_hash
                title = loaded.title
                document_version = document_hash[:16]
                if source_metadata.doi:
                    source_identity = f"doi:{source_metadata.doi.lower()}"
                elif source_metadata.source_url:
                    source_identity = f"url:{source_metadata.source_url}"
                else:
                    source_identity = f"path:{path}"
                document_id = _stable_id(source_identity, request.index_version, prefix="doc_")
                existing = self.get_document(document_id)
                if existing is None:
                    existing = self._document_by_source_path(str(path), request.index_version)
                    if existing is not None:
                        document_id = str(existing["document_id"])
                stored_ids: list[str] = []
                stored_split_versions: set[str] = set()
                if existing and existing.get("status") == "available":
                    stored = collection.get(
                        where=_chroma_where(request.index_version, document_id),
                        include=["metadatas"],
                    )
                    stored_ids = [str(value) for value in (stored.get("ids") or [])]
                    stored_split_versions = {
                        str(metadata.get("split_version") or "")
                        for metadata in (stored.get("metadatas") or [])
                    }
                split_is_current = bool(stored_ids) and stored_split_versions == {split_version}
                content_is_current = bool(existing) and existing.get("document_hash") == document_hash
                metadata_is_current = bool(existing) and all(
                    str(existing.get(key) or "") == value
                    for key, value in {
                        "title": title,
                        "source_path": str(path),
                        "source_type": loaded.source_type,
                        "corpus": source_metadata.corpus,
                        "source_url": source_metadata.source_url,
                        "doi": source_metadata.doi,
                        "license": source_metadata.license,
                        "metadata_json": json_dumps(source_metadata.metadata),
                    }.items()
                )
                if (
                    existing
                    and existing.get("status") == "available"
                    and split_is_current
                    and content_is_current
                    and metadata_is_current
                ):
                    results.append(
                        {
                            **existing,
                            "status": "available",
                            "warnings": loaded.warnings,
                            "idempotent": True,
                        }
                    )
                    continue

                chunks: list[tuple[str, str, int | None, int | None]] = []
                for section in loaded.sections:
                    section_chunks = self._split(
                        section.text,
                        chunk_size=request.chunk_size,
                        chunk_overlap=request.chunk_overlap,
                    )
                    chunks.extend(
                        (chunk, section.section, section.page_start, section.page_end)
                        for chunk in section_chunks
                    )
                now = time.time()
                ids: list[str] = []
                documents: list[str] = []
                metadatas: list[dict[str, Any]] = []
                db_rows: list[tuple[Any, ...]] = []
                for idx, (chunk, section, page_start, page_end) in enumerate(chunks):
                    text_hash = _sha256_text(chunk)
                    chunk_id = _stable_id(document_id, document_version, split_version, idx, text_hash, prefix="chk_")
                    ids.append(chunk_id)
                    documents.append(chunk)
                    metadata = {
                        "document_id": document_id,
                        "document_version": document_version,
                        "chunk_id": chunk_id,
                        "chunk_index": idx,
                        "title": title,
                        "source_path": str(path),
                        "source_type": loaded.source_type,
                        "corpus": source_metadata.corpus,
                        "source_url": source_metadata.source_url,
                        "doi": source_metadata.doi,
                        "license": source_metadata.license,
                        "section": section,
                        "text_hash": text_hash,
                        "index_version": request.index_version,
                        "split_version": split_version,
                    }
                    if page_start is not None:
                        metadata["page_start"] = page_start
                    if page_end is not None:
                        metadata["page_end"] = page_end
                    metadatas.append(metadata)
                    db_rows.append(
                        (
                            chunk_id,
                            document_id,
                            document_version,
                            idx,
                            title,
                            str(path),
                            section,
                            page_start,
                            page_end,
                            text_hash,
                            request.index_version,
                            chunk,
                            now,
                        )
                    )

                embeddings = self.embedding.embed_documents(documents)
                if ids:
                    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
                stale_ids = sorted(set(stored_ids).difference(ids))
                with connect(self.store.db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO yieldmind_documents
                            (document_id, title, source_path, source_type, corpus, source_url,
                             doi, license, metadata_json, document_hash, document_version,
                             status, index_version, chunk_count, error, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(document_id) DO UPDATE SET
                            title=excluded.title,
                            source_path=excluded.source_path,
                            source_type=excluded.source_type,
                            corpus=excluded.corpus,
                            source_url=excluded.source_url,
                            doi=excluded.doi,
                            license=excluded.license,
                            metadata_json=excluded.metadata_json,
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
                            loaded.source_type,
                            source_metadata.corpus,
                            source_metadata.source_url,
                            source_metadata.doi,
                            source_metadata.license,
                            json_dumps(source_metadata.metadata),
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
                             source_path, section, page_start, page_end, text_hash,
                             index_version, text, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(chunk_id) DO UPDATE SET
                            document_id=excluded.document_id,
                            document_version=excluded.document_version,
                            chunk_index=excluded.chunk_index,
                            title=excluded.title,
                            source_path=excluded.source_path,
                            section=excluded.section,
                            page_start=excluded.page_start,
                            page_end=excluded.page_end,
                            text_hash=excluded.text_hash,
                            index_version=excluded.index_version,
                            text=excluded.text,
                            created_at=excluded.created_at
                        """,
                        db_rows,
                    )
                # PostgreSQL/SQLite is the retrieval source of truth. Delete old
                # vectors only after the replacement chunk rows have committed.
                if stale_ids:
                    collection.delete(ids=stale_ids)
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
                    | {
                        "source_type": loaded.source_type,
                        "corpus": source_metadata.corpus,
                        "source_url": source_metadata.source_url,
                        "doi": source_metadata.doi,
                        "license": source_metadata.license,
                        "warnings": loaded.warnings,
                        "duration_seconds": round(time.time() - started, 4),
                        "idempotent": False,
                    }
                )
            except Exception as exc:
                results.append({"source_path": str(path), "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return {
            "index_version": request.index_version,
            "collection": self.collection_name,
            "embedding_profile": self.embedding_profile.model_dump(mode="json"),
            "embedding_profile_fingerprint": self.embedding_profile.fingerprint(),
            "embedding_model": self.embedding_profile.model_id,
            "split_version": split_version,
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
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
        documents = []
        for row in rows:
            document = dict(row)
            document["metadata"] = json_loads(document.pop("metadata_json", "{}"))
            documents.append(document)
        return documents

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
            SELECT c.*, d.source_type, d.corpus, d.source_url, d.doi, d.license
            FROM yieldmind_document_chunks c
            JOIN yieldmind_documents d ON d.document_id=c.document_id
            WHERE c.index_version=? AND d.status='available'
        """
        params: list[Any] = [request.index_version]
        if request.document_id:
            sql += " AND c.document_id=?"
            params.append(request.document_id)
        if request.corpus:
            sql += " AND d.corpus=?"
            params.append(request.corpus)
        sql += " ORDER BY c.document_id, c.chunk_index"
        with connect(self.store.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def _active_chunks_by_id(
        self,
        chunk_ids: list[str],
        request: KnowledgeSearchRequest,
    ) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        placeholders = ", ".join("?" for _ in chunk_ids)
        sql = f"""
            SELECT c.*, d.source_type, d.corpus, d.source_url, d.doi, d.license
            FROM yieldmind_document_chunks c
            JOIN yieldmind_documents d ON d.document_id=c.document_id
            WHERE c.chunk_id IN ({placeholders})
              AND c.index_version=? AND d.status='available'
        """
        params: list[Any] = [*chunk_ids, request.index_version]
        if request.document_id:
            sql += " AND c.document_id=?"
            params.append(request.document_id)
        if request.corpus:
            sql += " AND d.corpus=?"
            params.append(request.corpus)
        with connect(self.store.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return {str(row["chunk_id"]): dict(row) for row in rows}

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
            "source_type": metadata.get("source_type"),
            "corpus": metadata.get("corpus"),
            "source_url": metadata.get("source_url"),
            "doi": metadata.get("doi"),
            "license": metadata.get("license"),
            "section": metadata.get("section"),
            "page_start": metadata.get("page_start"),
            "page_end": metadata.get("page_end"),
            "chunk_index": metadata.get("chunk_index"),
            "text_hash": metadata.get("text_hash"),
            "index_version": metadata.get("index_version"),
            "score": score,
            "distance": distance,
            "text": text,
        }

    def _vector_hits(self, request: KnowledgeSearchRequest, *, candidate_k: int) -> list[dict[str, Any]]:
        collection = self._collection()
        where = _chroma_where(request.index_version, request.document_id, request.corpus)
        query_embedding = self.embedding.embed_query(request.query)
        try:
            collection_count = max(1, int(collection.count()))
        except Exception:
            collection_count = candidate_k * 3
        # Interrupted writes or a mistakenly shared Chroma directory can leave
        # vectors that are not active in this SQL store. Expand only when those
        # rows starve the requested committed result count.
        n_results = min(max(candidate_k * 3, candidate_k), collection_count)
        while True:
            raw = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            hits = []
            ids = (raw.get("ids") or [[]])[0]
            distances = (raw.get("distances") or [[]])[0]
            active_chunks = self._active_chunks_by_id([str(chunk_id) for chunk_id in ids], request)
            for chunk_id, distance in zip(ids, distances):
                committed = active_chunks.get(str(chunk_id))
                if committed is None:
                    continue
                score = 1.0 - float(distance) if distance is not None else None
                hits.append(
                    self._hit_from_metadata(
                        chunk_id=chunk_id,
                        text=str(committed["text"]),
                        metadata=committed,
                        score=score,
                        distance=float(distance) if distance is not None else None,
                    )
                )
                if len(hits) >= candidate_k:
                    break
            if len(hits) >= candidate_k or n_results >= collection_count:
                return hits
            n_results = min(collection_count, max(n_results + 1, n_results * 2))

    def _bm25_hits(self, request: KnowledgeSearchRequest, *, candidate_k: int) -> list[dict[str, Any]]:
        from rank_bm25 import BM25Plus

        chunks = self._active_chunks(request)
        if not chunks:
            return []
        tokenized_corpus = [self._lexical_tokens(str(chunk["text"])) for chunk in chunks]
        query_tokens = self._lexical_tokens(request.query)
        if not query_tokens:
            return []
        query_token_set = set(query_tokens)
        has_lexical_overlap = [bool(query_token_set.intersection(tokens)) for tokens in tokenized_corpus]
        scores = BM25Plus(tokenized_corpus).get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))
        hits: list[dict[str, Any]] = []
        for index, score in ranked:
            # BM25Plus adds delta even when a document has zero term overlap.
            # Such rows are not lexical candidates and must not enter RRF.
            if not has_lexical_overlap[index] or float(score) <= 0:
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
        channel_ranks: dict[str, dict[str, int]] = {}
        for channel, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
            for rank, hit in enumerate(hits, start=1):
                chunk_id = str(hit["chunk_id"])
                by_id.setdefault(chunk_id, hit.copy())
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
                channels.setdefault(chunk_id, []).append(channel)
                channel_ranks.setdefault(chunk_id, {})[channel] = rank

        def rank_key(chunk_id: str) -> tuple[Any, ...]:
            ranks = list(channel_ranks[chunk_id].values())
            hit = by_id[chunk_id]
            return (
                -scores[chunk_id],
                -len(ranks),
                min(ranks),
                sum(ranks),
                str(hit.get("source_path") or ""),
                int(hit.get("chunk_index") or 0),
                chunk_id,
            )

        ranked_ids = sorted(scores, key=rank_key)[:top_k]
        result = []
        for chunk_id in ranked_ids:
            hit = by_id[chunk_id]
            hit["score"] = scores[chunk_id]
            hit["distance"] = None
            hit["retrieval_channels"] = channels[chunk_id]
            hit["retrieval_channel_ranks"] = channel_ranks[chunk_id]
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
    def report_source_path(value: Any) -> str:
        raw_source = str(value or "")
        if not raw_source:
            return ""
        source = Path(raw_source)
        try:
            return str(source.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            return source.name

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
        source_match_rank = None
        terms_match_rank = None
        candidate_diagnostics = []
        for idx, hit in enumerate(result["hits"], start=1):
            text = str(hit.get("text") or "").lower()
            source_path = str(hit.get("source_path") or "")
            source_matches = not expected_sources or any(
                source_path == expected or source_path.endswith(f"/{expected}") for expected in expected_sources
            )
            matched_terms = [term for term in expected_terms if term in text]
            missing_terms = [term for term in expected_terms if term not in text]
            terms_match = not missing_terms
            if source_matches and source_match_rank is None:
                source_match_rank = idx
            if terms_match and terms_match_rank is None:
                terms_match_rank = idx
            candidate_diagnostics.append(
                {
                    "rank": idx,
                    "chunk_id": str(hit.get("chunk_id") or ""),
                    "source_path": report_source_path(source_path),
                    "chunk_index": hit.get("chunk_index"),
                    "source_matches": source_matches,
                    "terms_match": terms_match,
                    "matched_terms": matched_terms,
                    "missing_terms": missing_terms,
                    "score": hit.get("score"),
                    "retrieval_channels": hit.get("retrieval_channels", []),
                    "retrieval_channel_ranks": hit.get("retrieval_channel_ranks", {}),
                }
            )
            if source_matches and terms_match and rank is None:
                rank = idx
        if rank is not None:
            failure_reason = None if rank == 1 else "relevant_chunk_ranked_below_top_1"
        elif source_match_rank is not None:
            failure_reason = "expected_source_retrieved_but_terms_not_colocated"
        else:
            failure_reason = "expected_source_not_retrieved"
        rows.append(
            {
                "id": case.get("id", ""),
                "query": case["query"],
                "expected_source_paths": sorted(expected_sources),
                "expected_terms": expected_terms,
                "rank": rank,
                "passed": rank is not None,
                "failure_reason": failure_reason,
                "source_match_rank": source_match_rank,
                "terms_match_rank": terms_match_rank,
                "top_source_path": (
                    report_source_path(result["hits"][0].get("source_path")) if result["hits"] else None
                ),
                "candidates": candidate_diagnostics,
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
