from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from pydantic import ValidationError

import yieldmind.literature as literature
from yieldmind.literature import (
    LiteratureManifest,
    build_literature_ingest_request,
    download_literature,
)


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, url: str) -> None:
        super().__init__(payload)
        self._url = url

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _manifest(payload: bytes, *, expected_sha256: str | None = None) -> LiteratureManifest:
    return LiteratureManifest.model_validate(
        {
            "manifest_version": "1",
            "documents": [
                {
                    "source_id": "paper_1",
                    "title": "Open yield-stress paper",
                    "filename": "paper_1.pdf",
                    "download_url": "https://downloads.example.org/paper.pdf",
                    "source_url": "https://example.org/paper",
                    "doi": "https://doi.org/10.1000/example",
                    "license": "CC-BY-4.0",
                    "topics": ["yield_stress", "packing"],
                    "expected_sha256": expected_sha256 or hashlib.sha256(payload).hexdigest(),
                }
            ],
        }
    )


def test_download_literature_verifies_hash_and_reuses_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"%PDF-1.4\ncontrolled fixture"
    manifest = _manifest(payload)
    monkeypatch.setattr(literature, "_validate_public_https_url", lambda _: None)
    calls = 0

    def opener(request: object, *, timeout: float) -> _FakeResponse:
        nonlocal calls
        calls += 1
        assert timeout == 10
        return _FakeResponse(payload, "https://cdn.example.org/paper.pdf")

    first = download_literature(manifest, tmp_path, timeout_seconds=10, opener=opener)
    second = download_literature(manifest, tmp_path, timeout_seconds=10, opener=opener)

    assert first[0]["status"] == "downloaded"
    assert second[0]["status"] == "verified_existing"
    assert calls == 1
    assert (tmp_path / "paper_1.pdf").read_bytes() == payload


def test_download_literature_rejects_hash_mismatch_and_removes_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"%PDF-1.4\nchanged content"
    manifest = _manifest(payload, expected_sha256="0" * 64)
    monkeypatch.setattr(literature, "_validate_public_https_url", lambda _: None)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_literature(
            manifest,
            tmp_path,
            opener=lambda *_args, **_kwargs: _FakeResponse(payload, "https://example.org/paper.pdf"),
        )

    assert not (tmp_path / "paper_1.pdf").exists()
    assert not (tmp_path / "paper_1.pdf.part").exists()


def test_public_url_validation_rejects_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        literature.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("127.0.0.1", 443))],
    )

    with pytest.raises(ValueError, match="non-public"):
        literature._validate_public_https_url("https://localhost/paper.pdf")


def test_manifest_rejects_duplicate_doi() -> None:
    payload = b"%PDF-1.4\nfixture"
    source = _manifest(payload).documents[0].model_dump()
    duplicate = {**source, "source_id": "paper_2", "filename": "paper_2.pdf"}

    with pytest.raises(ValidationError, match="duplicate literature doi"):
        LiteratureManifest.model_validate({"documents": [source, duplicate]})


def test_build_ingest_request_carries_curated_metadata(tmp_path: Path) -> None:
    manifest = _manifest(b"%PDF-1.4\nfixture")

    request = build_literature_ingest_request(
        manifest,
        tmp_path,
        index_version="test-index-v1",
    )

    path = str((tmp_path / "paper_1.pdf").resolve())
    assert request.paths == [path]
    assert request.source_metadata[path].corpus == "literature"
    assert request.source_metadata[path].doi == "10.1000/example"
    assert request.source_metadata[path].metadata["topics"] == ["yield_stress", "packing"]
