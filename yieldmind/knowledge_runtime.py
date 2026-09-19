"""Server-controlled knowledge-base runtime selection."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from yieldmind.database import YieldMindStore
from yieldmind.knowledge_base import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    HttpEmbeddingFunction,
    KnowledgeBase,
    qwen3_embedding_profile,
)


class KnowledgeRuntimeConfig(BaseModel):
    profile: Literal["local_hashing", "qwen3-embedding-0.6b"] = "local_hashing"
    embedding_endpoint: str = ""
    embedding_token_env: str = "YIELDMIND_EMBEDDING_TOKEN"
    embedding_timeout_seconds: float = Field(default=300.0, gt=0.0, le=900.0)
    embedding_batch_size: int = Field(default=8, ge=1, le=64)
    chroma_dir: str = str(DEFAULT_CHROMA_DIR)
    collection_name: str = DEFAULT_COLLECTION

    @model_validator(mode="after")
    def validate_qwen_endpoint(self) -> "KnowledgeRuntimeConfig":
        if self.profile == "qwen3-embedding-0.6b" and not self.embedding_endpoint:
            raise ValueError("Qwen3 runtime requires YIELDMIND_EMBEDDING_ENDPOINT.")
        return self

    @classmethod
    def from_env(cls) -> "KnowledgeRuntimeConfig":
        return cls(
            profile=os.environ.get("YIELDMIND_EMBEDDING_PROFILE", "local_hashing"),
            embedding_endpoint=os.environ.get("YIELDMIND_EMBEDDING_ENDPOINT", ""),
            embedding_token_env=os.environ.get("YIELDMIND_EMBEDDING_TOKEN_ENV", "YIELDMIND_EMBEDDING_TOKEN"),
            embedding_timeout_seconds=float(os.environ.get("YIELDMIND_EMBEDDING_TIMEOUT_SECONDS", "300")),
            embedding_batch_size=int(os.environ.get("YIELDMIND_EMBEDDING_BATCH_SIZE", "8")),
            chroma_dir=os.environ.get("YIELDMIND_CHROMA_DIR", str(DEFAULT_CHROMA_DIR)),
            collection_name=os.environ.get("YIELDMIND_CHROMA_COLLECTION", DEFAULT_COLLECTION),
        )

    def public_summary(self) -> dict[str, Any]:
        profile = qwen3_embedding_profile(http=True) if self.profile == "qwen3-embedding-0.6b" else None
        return {
            "profile": self.profile,
            "embedding_endpoint_configured": bool(self.embedding_endpoint),
            "embedding_token_configured": bool(os.environ.get(self.embedding_token_env, "")),
            "embedding_timeout_seconds": self.embedding_timeout_seconds,
            "embedding_batch_size": self.embedding_batch_size,
            "chroma_dir_configured": bool(self.chroma_dir),
            "collection_name": self.collection_name,
            "embedding_profile": profile.model_dump(mode="json") if profile else None,
        }


def configured_knowledge_base(
    store: YieldMindStore,
    *,
    config: KnowledgeRuntimeConfig | None = None,
) -> KnowledgeBase:
    runtime = config or KnowledgeRuntimeConfig.from_env()
    if runtime.profile == "local_hashing":
        return KnowledgeBase(
            store=store,
            chroma_dir=runtime.chroma_dir,
            collection_name=runtime.collection_name,
        )
    profile = qwen3_embedding_profile(http=True)
    embedding = HttpEmbeddingFunction(
        profile,
        endpoint=runtime.embedding_endpoint,
        token=os.environ.get(runtime.embedding_token_env, ""),
        timeout_seconds=runtime.embedding_timeout_seconds,
        batch_size=runtime.embedding_batch_size,
    )
    return KnowledgeBase(
        store=store,
        chroma_dir=runtime.chroma_dir,
        collection_name=runtime.collection_name,
        embedding=embedding,
        embedding_profile=profile,
    )


def knowledge_runtime_health() -> dict[str, Any]:
    try:
        config = KnowledgeRuntimeConfig.from_env()
        summary = config.public_summary()
        if config.profile == "local_hashing":
            return {"ok": True, **summary, "mode": "offline_deterministic_hashing"}
        profile = qwen3_embedding_profile(http=True)
        embedding = HttpEmbeddingFunction(
            profile,
            endpoint=config.embedding_endpoint,
            token=os.environ.get(config.embedding_token_env, ""),
            timeout_seconds=min(config.embedding_timeout_seconds, 10.0),
            batch_size=config.embedding_batch_size,
        )
        service = embedding.health()
        return {
            "ok": True,
            **summary,
            "mode": "isolated_http_sentence_transformer_real_inference",
            "service": {
                key: service.get(key)
                for key in (
                    "model_id",
                    "revision",
                    "dimensions",
                    "device",
                    "versions",
                    "process_peak_rss_bytes",
                    "encode_calls",
                    "encoded_texts",
                )
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "profile": os.environ.get("YIELDMIND_EMBEDDING_PROFILE", "local_hashing"),
            "error": f"{type(exc).__name__}: {exc}",
        }
