"""Manifest-driven, reproducible acquisition of open literature documents."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator, model_validator

from yieldmind.document_loader import MAX_DOCUMENT_BYTES, SUPPORTED_DOCUMENT_SUFFIXES
from yieldmind.knowledge_base import KnowledgeIngestRequest, KnowledgeSourceMetadata


USER_AGENT = "YieldMind-Literature/1.0 (+local reproducible research ingestion)"


class LiteratureSource(BaseModel):
    source_id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]+$")
    title: str = Field(..., min_length=3)
    filename: str
    download_url: str
    source_url: str
    doi: str = ""
    license: str = Field(..., min_length=2)
    topics: list[str] = Field(default_factory=list)
    expected_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")

    @field_validator("download_url", "source_url")
    @classmethod
    def validate_https_url(cls, value: str) -> str:
        parsed = urllib.parse.urlparse(value.strip())
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("literature URLs must be credential-free HTTPS URLs")
        return value.strip()

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        filename = value.strip()
        if Path(filename).name != filename:
            raise ValueError("filename must not contain a directory")
        if Path(filename).suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
            raise ValueError(f"unsupported literature document suffix: {Path(filename).suffix}")
        return filename

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str) -> str:
        return value.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")


class LiteratureManifest(BaseModel):
    manifest_version: str = "1"
    documents: list[LiteratureSource] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_unique_identity(self) -> "LiteratureManifest":
        for field in ("source_id", "filename"):
            values = [getattr(document, field) for document in self.documents]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate literature {field}")
        dois = [document.doi.lower() for document in self.documents if document.doi]
        if len(dois) != len(set(dois)):
            raise ValueError("duplicate literature doi")
        return self


def load_literature_manifest(path: str | Path) -> LiteratureManifest:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return LiteratureManifest.model_validate(payload)


def _validate_public_https_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("download URL must use HTTPS")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError(f"download URL resolves to a non-public address: {parsed.hostname}")


class _PublicHttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        _validate_public_https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_downloaded_content(path: Path) -> None:
    prefix = path.read_bytes()[:512].lstrip()
    suffix = path.suffix.lower()
    if suffix == ".pdf" and not prefix.startswith(b"%PDF-"):
        raise ValueError(f"downloaded content is not a PDF: {path.name}")
    if suffix in {".html", ".htm"} and b"<html" not in prefix.lower() and b"<!doctype html" not in prefix.lower():
        raise ValueError(f"downloaded content is not HTML: {path.name}")
    if suffix == ".docx" and not prefix.startswith(b"PK"):
        raise ValueError(f"downloaded content is not a DOCX archive: {path.name}")


def download_literature(
    manifest: LiteratureManifest,
    output_dir: str | Path,
    *,
    timeout_seconds: float = 60.0,
    opener: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    open_url = opener
    if open_url is None:
        open_url = urllib.request.build_opener(_PublicHttpsRedirectHandler()).open

    results: list[dict[str, Any]] = []
    for source in manifest.documents:
        target = destination / source.filename
        if target.exists():
            current_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            if current_hash == source.expected_sha256:
                results.append(
                    {
                        "source_id": source.source_id,
                        "path": str(target),
                        "sha256": current_hash,
                        "status": "verified_existing",
                    }
                )
                continue

        _validate_public_https_url(source.download_url)
        request = urllib.request.Request(source.download_url, headers={"User-Agent": USER_AGENT})
        temporary = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        size = 0
        try:
            with open_url(request, timeout=timeout_seconds) as response, temporary.open("wb") as handle:
                final_url = str(response.geturl())
                _validate_public_https_url(final_url)
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > MAX_DOCUMENT_BYTES:
                        raise ValueError(
                            f"literature document exceeds {MAX_DOCUMENT_BYTES} bytes: {source.source_id}"
                        )
                    digest.update(block)
                    handle.write(block)
            actual_hash = digest.hexdigest()
            if actual_hash != source.expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {source.source_id}: {actual_hash} != {source.expected_sha256}"
                )
            _validate_downloaded_content(temporary)
            os.replace(temporary, target)
            results.append(
                {
                    "source_id": source.source_id,
                    "path": str(target),
                    "sha256": actual_hash,
                    "bytes": size,
                    "status": "downloaded",
                }
            )
        finally:
            temporary.unlink(missing_ok=True)
    return results


def build_literature_ingest_request(
    manifest: LiteratureManifest,
    document_dir: str | Path,
    *,
    index_version: str,
    chunk_size: int = 700,
    chunk_overlap: int = 80,
) -> KnowledgeIngestRequest:
    root = Path(document_dir).expanduser().resolve()
    paths = [str(root / source.filename) for source in manifest.documents]
    metadata = {
        str(root / source.filename): KnowledgeSourceMetadata(
            corpus="literature",
            title=source.title,
            source_url=source.source_url,
            doi=source.doi,
            license=source.license,
            metadata={"source_id": source.source_id, "topics": source.topics},
        )
        for source in manifest.documents
    }
    return KnowledgeIngestRequest(
        paths=paths,
        index_version=index_version,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_metadata=metadata,
    )
