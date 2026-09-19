"""Format-aware local document loading for the YieldMind knowledge base."""

from __future__ import annotations

import hashlib
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_DOCUMENT_SUFFIXES = {".docx", ".htm", ".html", ".md", ".pdf", ".txt"}
MAX_DOCUMENT_BYTES = 100 * 1024 * 1024
MAX_PDF_PAGES = 2_000
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PDF_PAGE_MARKER_RE = re.compile(r"\b\d+\s+(?:of|/)\s+\d+\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class DocumentSection:
    text: str
    section: str = ""
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    title: str
    source_type: str
    document_hash: str
    sections: list[DocumentSection]
    warnings: list[str] = field(default_factory=list)


def _clean_text(text: str) -> str:
    lines = [line.replace("\x00", "").rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _clean_pdf_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    return _clean_text("\n".join(lines))


def _strip_repeated_pdf_edges(page_texts: list[str]) -> list[str]:
    edge_counts: dict[str, int] = {}
    page_lines: list[list[str]] = []
    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        page_lines.append(lines)
        if len(lines) >= 6:
            for line in set(lines[:2] + lines[-2:]):
                edge_counts[line] = edge_counts.get(line, 0) + 1
    threshold = max(3, math.ceil(len(page_lines) * 0.5))
    repeated = {line for line, count in edge_counts.items() if count >= threshold}

    cleaned_pages = []
    for lines in page_lines:
        last_edge_start = max(0, len(lines) - 2)
        kept = []
        for index, line in enumerate(lines):
            at_edge = index < 2 or index >= last_edge_start
            repeated_edge = len(lines) >= 6 and at_edge and line in repeated
            if repeated_edge or (at_edge and _PDF_PAGE_MARKER_RE.search(line)):
                continue
            kept.append(line)
        cleaned_pages.append(_clean_text("\n".join(kept)))
    return cleaned_pages


def _decode_text(data: bytes, path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode document with utf-8/gb18030/gbk: {path}")


def _markdown_sections(text: str) -> tuple[list[DocumentSection], str]:
    sections: list[DocumentSection] = []
    current_lines: list[str] = []
    heading_stack: list[tuple[int, str]] = []
    first_heading = ""

    def flush() -> None:
        cleaned = _clean_text("\n".join(current_lines))
        if cleaned:
            section = " > ".join(value for _, value in heading_stack)
            sections.append(DocumentSection(text=cleaned, section=section))

    for line in text.splitlines():
        match = _MARKDOWN_HEADING_RE.match(line)
        if match:
            flush()
            current_lines = [line]
            level = len(match.group(1))
            heading = match.group(2).strip()
            if not first_heading:
                first_heading = heading
            heading_stack = [(depth, value) for depth, value in heading_stack if depth < level]
            heading_stack.append((level, heading))
        else:
            current_lines.append(line)
    flush()
    return sections, first_heading


def _load_markdown_or_text(path: Path, data: bytes) -> tuple[list[DocumentSection], str, list[str]]:
    text = _decode_text(data, path)
    if path.suffix.lower() == ".md":
        sections, title = _markdown_sections(text)
    else:
        cleaned = _clean_text(text)
        sections = [DocumentSection(text=cleaned)] if cleaned else []
        title = ""
    return sections, title, []


def _load_pdf(path: Path, data: bytes) -> tuple[list[DocumentSection], str, list[str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf; install the project requirements.") from exc

    reader = PdfReader(io.BytesIO(data), strict=False)
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"Encrypted PDF cannot be opened without a password: {path}") from exc
        if not unlocked:
            raise ValueError(f"Encrypted PDF cannot be opened without a password: {path}")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDF has {len(reader.pages)} pages; maximum is {MAX_PDF_PAGES}: {path}")

    raw_pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            if "/Contents" not in page:
                text = ""
            else:
                text = _clean_pdf_text(page.extract_text(extraction_mode="layout") or "")
        except Exception as exc:
            raise ValueError(f"Could not extract PDF page {page_number}: {path}") from exc
        raw_pages.append(text)

    cleaned_pages = _strip_repeated_pdf_edges(raw_pages)
    sections: list[DocumentSection] = []
    blank_pages: list[int] = []
    for page_number, text in enumerate(cleaned_pages, start=1):
        if text:
            sections.append(DocumentSection(text=text, page_start=page_number, page_end=page_number))
        else:
            blank_pages.append(page_number)

    if not sections:
        raise ValueError(f"PDF contains no extractable text and may require OCR: {path}")
    warnings = []
    if blank_pages:
        preview = ", ".join(str(page) for page in blank_pages[:10])
        suffix = "..." if len(blank_pages) > 10 else ""
        warnings.append(f"Skipped {len(blank_pages)} PDF page(s) without extractable text: {preview}{suffix}")
    metadata = reader.metadata or {}
    title = _clean_text(str(metadata.get("/Title") or ""))
    return sections, title, warnings


def _load_docx(path: Path, data: bytes) -> tuple[list[DocumentSection], str, list[str]]:
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise RuntimeError("DOCX ingestion requires python-docx; install the project requirements.") from exc

    document = Document(io.BytesIO(data))
    sections: list[DocumentSection] = []
    current_lines: list[str] = []
    heading_stack: list[tuple[int, str]] = []
    first_heading = ""

    def flush() -> None:
        cleaned = _clean_text("\n".join(current_lines))
        if cleaned:
            section = " > ".join(value for _, value in heading_stack)
            sections.append(DocumentSection(text=cleaned, section=section))

    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            text = _clean_text(block.text)
            if not text:
                continue
            style_name = str(block.style.name or "") if block.style else ""
            heading_match = re.match(r"Heading\s+(\d+)", style_name, flags=re.IGNORECASE)
            if heading_match:
                flush()
                current_lines = [text]
                level = int(heading_match.group(1))
                if not first_heading:
                    first_heading = text
                heading_stack = [(depth, value) for depth, value in heading_stack if depth < level]
                heading_stack.append((level, text))
            else:
                current_lines.append(text)
        elif isinstance(block, Table):
            rows = []
            for row in block.rows:
                cells = [_clean_text(cell.text) for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                current_lines.extend(rows)
    flush()
    return sections, first_heading, []


def _load_html(path: Path, data: bytes) -> tuple[list[DocumentSection], str, list[str]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("HTML ingestion requires beautifulsoup4; install the project requirements.") from exc

    soup = BeautifulSoup(_decode_text(data, path), "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    sections: list[DocumentSection] = []
    current_lines: list[str] = []
    heading_stack: list[tuple[int, str]] = []
    first_heading = ""

    def flush() -> None:
        cleaned = _clean_text("\n".join(current_lines))
        if cleaned:
            section = " > ".join(value for _, value in heading_stack)
            sections.append(DocumentSection(text=cleaned, section=section))

    root = soup.find("article") or soup.find("main") or soup.body or soup
    for element in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "tr"]):
        text = _clean_text(element.get_text(" ", strip=True))
        if not text:
            continue
        if element.name and element.name.startswith("h"):
            flush()
            current_lines = [text]
            level = int(element.name[1])
            if not first_heading:
                first_heading = text
            heading_stack = [(depth, value) for depth, value in heading_stack if depth < level]
            heading_stack.append((level, text))
        else:
            current_lines.append(text)
    flush()
    html_title = _clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    return sections, first_heading or html_title, []


def load_document(path: str | Path, *, title: str = "") -> LoadedDocument:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Document path does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Document path is not a file: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValueError(
            f"Unsupported document type {suffix!r}; supported: {sorted(SUPPORTED_DOCUMENT_SUFFIXES)}"
        )
    size = resolved.stat().st_size
    if size > MAX_DOCUMENT_BYTES:
        raise ValueError(f"Document is {size} bytes; maximum is {MAX_DOCUMENT_BYTES}: {resolved}")
    data = resolved.read_bytes()
    document_hash = hashlib.sha256(data).hexdigest()

    if suffix in {".md", ".txt"}:
        sections, extracted_title, warnings = _load_markdown_or_text(resolved, data)
    elif suffix == ".pdf":
        sections, extracted_title, warnings = _load_pdf(resolved, data)
    elif suffix == ".docx":
        sections, extracted_title, warnings = _load_docx(resolved, data)
    else:
        sections, extracted_title, warnings = _load_html(resolved, data)
    if not sections:
        raise ValueError(f"Document contains no extractable text: {resolved}")

    return LoadedDocument(
        path=resolved,
        title=title.strip() or extracted_title or resolved.stem,
        source_type=suffix.lstrip("."),
        document_hash=document_hash,
        sections=sections,
        warnings=warnings,
    )
