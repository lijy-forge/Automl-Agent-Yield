"""Unified external retrieval for yield-stress AutoML.

This module is intentionally yield-specific and does not depend on the legacy
Research Mode. It borrows the useful source schema idea: every source
gets a stable source_id, source_type, provider, query, title, link, and snippet.
"""

from __future__ import annotations

import time
from typing import Callable

import requests

from configs import Configs
from utils import search_web


YieldEmit = Callable[[str, str], None]


def _yield_web_queries(user_prompt: str = "") -> list[str]:
    queries = [
        "high solid suspension yield stress model packing fraction machine learning",
        "high solid slurry yield stress physics informed machine learning",
        "yield stress model suspensions YODEL Flatt Bowen 2006",
        "dense suspension yield stress model particle size packing fraction",
        "Herschel Bulkley Bingham yield stress prediction suspension slurry",
        "cement paste yield stress machine learning prediction rheology",
    ]
    if user_prompt:
        queries.append(f"{user_prompt} yield stress prediction rheology model")
    return queries


def _yield_arxiv_queries() -> list[str]:
    return [
        'search_query=all:"yield stress" AND all:"machine learning"',
        'search_query=all:"physics-informed" AND all:rheology',
        'search_query=all:suspension AND all:"yield stress"',
        'search_query=all:"high solid" AND all:"yield stress"',
        'search_query=all:"cement paste" AND all:rheology',
        'search_query=all:PINN AND all:rheology',
    ]


def _yield_scholar_queries() -> list[str]:
    return [
        "YODEL yield stress model suspensions",
        "high solid suspension yield stress machine learning prediction",
        "suspension rheology yield stress packing fraction",
        "dense slurry yield stress physics informed model",
        "physics informed neural network rheology yield stress",
        "Herschel Bulkley Bingham slurry suspension yield stress",
    ]


def _dedupe_texts(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_yield_query_plan(user_prompt: str = "", extra_queries: list[str] | None = None) -> list[dict]:
    """Return the grouped query plan used by the yield retriever."""
    web_queries = _yield_web_queries(user_prompt)
    if extra_queries:
        web_queries.extend(extra_queries)
    plan = [
        {
            "source": "Web Search",
            "provider": Configs.SEARCH_PROVIDER,
            "queries": _dedupe_texts(web_queries),
        },
        {
            "source": "arXiv",
            "provider": "arXiv",
            "queries": _yield_arxiv_queries(),
        },
        {
            "source": "Semantic Scholar",
            "provider": "Semantic Scholar",
            "queries": _yield_scholar_queries(),
        },
        {
            "source": "OpenAlex",
            "provider": "OpenAlex",
            "queries": _yield_scholar_queries(),
        },
        {
            "source": "Crossref",
            "provider": "Crossref",
            "queries": _yield_scholar_queries(),
        },
    ]
    if Configs.SERPAPI_API_KEY:
        plan.append(
            {
                "source": "SerpAPI Scholar",
                "provider": "SerpAPI Google Scholar",
                "queries": _yield_scholar_queries(),
            }
        )
    return plan


def _as_source(source_type: str, provider: str, query: str, title: str, link: str, snippet: str, **extra) -> dict:
    item = {
        "source_type": source_type,
        "provider": provider,
        "source": provider,
        "query": str(query or ""),
        "title": str(title or "").strip(),
        "link": str(link or "").strip(),
        "url": str(link or "").strip(),
        "snippet": str(snippet or "").strip(),
    }
    for key, value in extra.items():
        if value not in (None, ""):
            item[key] = value
    return item


def _classify_yield_source_relevance(item: dict) -> dict:
    """Deterministic relevance labels for yield-stress search evidence."""
    evidence_text = " ".join(
        str(item.get(key, "") or "")
        for key in ("title", "snippet", "venue", "doi")
    ).lower()
    query_text = str(item.get("query", "") or "").lower()
    text = f"{evidence_text} {query_text}"

    direct_domain_terms = (
        "yield stress",
        "yield-stress",
        "yielding stress",
        "屈服",
        "cement paste",
        "cementitious",
        "concrete",
        "grout",
        "slurry",
        "paste",
        "suspension",
        "dense suspension",
        "high solid",
        "high-solid",
    )
    mechanism_terms = (
        "yodel",
        "flatt",
        "bowen",
        "packing fraction",
        "maximum packing",
        "solid volume fraction",
        "volume fraction",
        "herschel",
        "bulkley",
        "bingham",
        "rheology",
        "rheological",
        "superplasticizer",
        "particle size",
    )
    method_terms = (
        "machine learning",
        "neural network",
        "physics-informed",
        "physics informed",
        "pinn",
        "gaussian process",
        "random forest",
        "gradient boosting",
        "xgboost",
        "symbolic regression",
        "multi-fidelity",
        "multifidelity",
    )
    weak_terms = (
        "tokamak",
        "hamilton",
        "bellman",
        "reinforcement learning",
        "sea ice",
        "land ice",
        "quantum",
        "wireless",
        "mimo",
        "edge computing",
        "microvascular",
        "structural stress",
        "full-field structural",
        "heterogeneous graph",
    )

    evidence_domain_hit = any(term in evidence_text for term in direct_domain_terms)
    evidence_mechanism_hit = any(term in evidence_text for term in mechanism_terms)
    evidence_method_hit = any(term in evidence_text for term in method_terms)
    query_domain_hit = any(term in query_text for term in direct_domain_terms)
    query_method_hit = any(term in query_text for term in method_terms)
    domain_hit = evidence_domain_hit
    mechanism_hit = evidence_mechanism_hit
    method_hit = evidence_method_hit
    weak_hit = any(term in evidence_text for term in weak_terms)

    domain_score = 5 if domain_hit else (3 if mechanism_hit else 1)
    method_score = 4 if method_hit else 1

    if domain_hit and mechanism_hit:
        label = "高"
        score = 5
        category = "屈服值/浆料流变/机理直接相关"
        reason = "同时命中屈服值、高固含浆料/悬浮液/水泥浆体和机理/流变关键词。"
    elif domain_hit:
        label = "高"
        score = 4
        category = "屈服值/浆料直接相关"
        reason = "直接命中 yield stress、high-solid slurry、suspension、cement paste 等任务关键词。"
    elif mechanism_hit and not weak_hit:
        label = "中"
        score = 3
        category = "可迁移流变机理"
        reason = "命中 packing、Bingham、Herschel-Bulkley、YODEL 或流变关键词，但任务场景不一定完全一致。"
    elif method_hit and not weak_hit:
        label = "中"
        score = 2
        category = "通用 ML/PINN 方法参考"
        reason = "主要提供建模方法启发，需要由屈服值数据和机理约束二次筛选。"
    elif (query_domain_hit or query_method_hit) and not weak_hit:
        label = "低"
        score = 1
        category = "查询词相关但证据文本未命中"
        reason = "相关性主要来自查询词，标题/摘要本身未命中屈服值、浆料或流变核心关键词。"
    else:
        label = "低"
        score = 1
        category = "弱相关或背景参考"
        reason = "未命中屈服值/浆料/流变核心关键词。"

    if weak_hit and not domain_hit:
        label = "低"
        score = min(score, 1)
        category = "弱相关外部物理/工程场景"
        reason = "主题偏其他物理或工程场景，只能作为弱方法参考。"

    if domain_hit and mechanism_hit:
        evidence_role = "mechanism_and_domain_support"
    elif domain_hit and method_hit:
        evidence_role = "domain_and_method_support"
    elif domain_hit or mechanism_hit:
        evidence_role = "domain_or_mechanism_support"
    elif method_hit:
        evidence_role = "method_support"
    else:
        evidence_role = "weak_reference"

    return {
        "relevance_label": label,
        "relevance_score": score,
        "domain_score": domain_score,
        "method_score": method_score,
        "evidence_role": evidence_role,
        "category": category,
        "relevance_reason": reason,
    }


def _with_relevance(item: dict) -> dict:
    enriched = dict(item or {})
    for key, value in _classify_yield_source_relevance(enriched).items():
        enriched.setdefault(key, value)
    return enriched


def _dedupe(sources: list[dict], limit: int = 40) -> list[dict]:
    seen = set()
    out = []
    for item in sources:
        title = str(item.get("title") or "").strip()
        link = str(item.get("link") or item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if not title and not link:
            continue
        text = f"{title} {snippet}".lower()
        if "no information is available for this page" in text or "没有此页面的相关信息" in text:
            continue
        key = link.lower().rstrip("/") or title.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(_with_relevance(item))
        if len(out) >= limit:
            break
    return out


def _number_sources(sources: list[dict]) -> list[dict]:
    numbered = []
    for idx, item in enumerate(sources, 1):
        enriched = dict(item)
        enriched["id"] = idx
        enriched["source_id"] = str(idx)
        numbered.append(enriched)
    return numbered


def _summary_by_key(sources: list[dict], key_name: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in sources:
        key = str(item.get(key_name) or "Unknown")
        summary[key] = summary.get(key, 0) + 1
    return summary


def _quality_summary(sources: list[dict]) -> dict:
    relevance_summary = {
        "high": sum(1 for item in sources if float(item.get("relevance_score", 0) or 0) >= 4),
        "medium": sum(1 for item in sources if 2 <= float(item.get("relevance_score", 0) or 0) < 4),
        "low": sum(1 for item in sources if float(item.get("relevance_score", 0) or 0) < 2),
    }
    return {
        "relevance_summary": relevance_summary,
        "quality_rule": (
            "Prefer yield-stress/cement-paste/slurry/suspension sources with "
            "relevance_score>=3. Treat generic PINN/ML papers as method support "
            "unless they also match yield-stress rheology terms."
        ),
    }


def retrieve_web_sources(queries: list[str], top_k_per_query: int, emit: YieldEmit | None = None) -> tuple[list[dict], list[dict]]:
    sources = []
    query_log = []
    for query in queries:
        query_log.append({"source": "Web Search", "provider": Configs.SEARCH_PROVIDER, "query": query})
        if emit:
            emit("search", f"Querying Web Search: {query}")
        try:
            results = search_web(query)[:top_k_per_query]
        except Exception as exc:
            if emit:
                emit("system", f"Web Search failed for query: {query}\n{type(exc).__name__}: {exc}")
            continue
        for item in results:
            provider = str(item.get("provider") or "Web Search")
            sources.append(
                _as_source(
                    "web",
                    provider,
                    query,
                    item.get("title", ""),
                    item.get("link") or item.get("url", ""),
                    item.get("snippet") or item.get("description", ""),
                )
            )
    return sources, query_log


def retrieve_arxiv_sources(top_k_per_query: int, emit: YieldEmit | None = None) -> tuple[list[dict], list[dict]]:
    try:
        import arxivloader
    except Exception as exc:
        if emit:
            emit("system", f"arXiv loader unavailable: {type(exc).__name__}: {exc}")
        return [], [{"source": "arXiv", "query": "", "error": "import_failed"}]

    sources = []
    query_log = []
    for query in _yield_arxiv_queries():
        query_log.append({"source": "arXiv", "query": query})
        if emit:
            emit("search", f"Querying arXiv: {query}")
        try:
            df = arxivloader.load(query, num=top_k_per_query, sortBy="submittedDate", verbosity=0)
        except Exception as exc:
            if emit:
                emit("system", f"arXiv query failed: {query}\n{type(exc).__name__}: {exc}")
            continue
        for _, row in df.iterrows():
            sources.append(
                _as_source(
                    "arxiv",
                    "arXiv",
                    query,
                    row.get("title", ""),
                    row.get("entry_id", row.get("url", row.get("pdf_url", ""))),
                    row.get("summary", row.get("abstract", "")),
                    authors=str(row.get("authors", "")),
                )
            )
    return sources, query_log


def retrieve_semantic_scholar_sources(top_k_per_query: int, emit: YieldEmit | None = None) -> tuple[list[dict], list[dict]]:
    sources = []
    query_log = []
    headers = {"User-Agent": "automl-agent-yield/1.0"}
    if Configs.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = Configs.SEMANTIC_SCHOLAR_API_KEY
    for query in _yield_scholar_queries():
        query_log.append({"source": "Semantic Scholar", "query": query})
        if emit:
            emit("search", f"Querying Semantic Scholar: {query}")
        for attempt in range(2):
            try:
                resp = requests.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": query,
                        "limit": top_k_per_query,
                        "fields": "title,abstract,authors,year,venue,url,externalIds,openAccessPdf,citationCount",
                    },
                    headers=headers,
                    timeout=15,
                )
                if resp.status_code == 429:
                    time.sleep(4 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    if emit:
                        emit("system", f"Semantic Scholar failed: HTTP {resp.status_code} for {query}")
                    break
                for paper in resp.json().get("data", []) or []:
                    authors = ", ".join(a.get("name", "") for a in paper.get("authors", []) or [])
                    external_ids = paper.get("externalIds") if isinstance(paper.get("externalIds"), dict) else {}
                    open_pdf = paper.get("openAccessPdf") if isinstance(paper.get("openAccessPdf"), dict) else {}
                    sources.append(
                        _as_source(
                            "semantic_scholar",
                            "Semantic Scholar",
                            query,
                            paper.get("title", ""),
                            paper.get("url", ""),
                            paper.get("abstract") or "",
                            authors=authors,
                            year=paper.get("year", ""),
                            venue=paper.get("venue", ""),
                            doi=external_ids.get("DOI", ""),
                            citation_count=paper.get("citationCount", 0),
                            open_access_pdf=open_pdf.get("url", ""),
                        )
                    )
                break
            except Exception as exc:
                if emit:
                    emit("system", f"Semantic Scholar request failed for {query}: {type(exc).__name__}: {exc}")
                break
    return sources, query_log


def retrieve_openalex_sources(top_k_per_query: int, emit: YieldEmit | None = None) -> tuple[list[dict], list[dict]]:
    sources = []
    query_log = []
    headers = {"User-Agent": "automl-agent-yield/1.0"}
    for query in _yield_scholar_queries():
        query_log.append({"source": "OpenAlex", "query": query})
        if emit:
            emit("search", f"Querying OpenAlex: {query}")
        try:
            params = {
                "search": query,
                "per-page": top_k_per_query,
                "select": "id,display_name,doi,publication_year,cited_by_count,primary_location",
            }
            if Configs.OPENALEX_API_KEY:
                params["api_key"] = Configs.OPENALEX_API_KEY
            if Configs.OPENALEX_EMAIL:
                params["mailto"] = Configs.OPENALEX_EMAIL
            resp = requests.get(
                "https://api.openalex.org/works",
                params=params,
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                if emit:
                    emit("system", f"OpenAlex failed: HTTP {resp.status_code} for {query}")
                continue
            for work in resp.json().get("results", []) or []:
                primary = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
                venue = ""
                if isinstance(primary.get("source"), dict):
                    venue = primary["source"].get("display_name", "")
                sources.append(
                    _as_source(
                        "openalex",
                        "OpenAlex",
                        query,
                        work.get("display_name", ""),
                        work.get("doi") or work.get("id", ""),
                        "",
                        year=work.get("publication_year", ""),
                        venue=venue,
                        cited_by_count=work.get("cited_by_count", 0),
                        doi=work.get("doi", ""),
                    )
                )
        except Exception as exc:
            if emit:
                emit("system", f"OpenAlex request failed for {query}: {type(exc).__name__}: {exc}")
    return sources, query_log


def _first_crossref_value(value) -> str:
    if isinstance(value, list) and value:
        return str(value[0] or "")
    return str(value or "")


def _crossref_year(item: dict) -> str:
    for key in ("published-print", "published-online", "published", "issued"):
        date_parts = item.get(key, {}).get("date-parts") if isinstance(item.get(key), dict) else None
        if date_parts and isinstance(date_parts, list) and date_parts[0]:
            return str(date_parts[0][0])
    return ""


def retrieve_crossref_sources(top_k_per_query: int, emit: YieldEmit | None = None) -> tuple[list[dict], list[dict]]:
    sources = []
    query_log = []
    for query in _yield_scholar_queries():
        query_log.append({"source": "Crossref", "query": query})
        if emit:
            emit("search", f"Querying Crossref: {query}")
        try:
            params = {
                "query.bibliographic": query,
                "rows": top_k_per_query,
                "select": "DOI,title,author,published-print,published-online,published,issued,container-title,URL,abstract,is-referenced-by-count,subject",
            }
            if Configs.CROSSREF_MAILTO:
                params["mailto"] = Configs.CROSSREF_MAILTO
            resp = requests.get(
                "https://api.crossref.org/works",
                params=params,
                headers={"User-Agent": "automl-agent-yield/1.0"},
                timeout=15,
            )
            if resp.status_code != 200:
                if emit:
                    emit("system", f"Crossref failed: HTTP {resp.status_code} for {query}")
                continue
            for work in (resp.json().get("message") or {}).get("items", []) or []:
                author_names = []
                for author in work.get("author", []) or []:
                    given = author.get("given", "")
                    family = author.get("family", "")
                    name = " ".join(x for x in (given, family) if x).strip()
                    if name:
                        author_names.append(name)
                doi = str(work.get("DOI") or "")
                link = str(work.get("URL") or "")
                if doi and not link:
                    link = f"https://doi.org/{doi}"
                sources.append(
                    _as_source(
                        "crossref",
                        "Crossref",
                        query,
                        _first_crossref_value(work.get("title")),
                        link,
                        work.get("abstract") or "",
                        authors=", ".join(author_names),
                        year=_crossref_year(work),
                        venue=_first_crossref_value(work.get("container-title")),
                        doi=doi,
                        cited_by_count=work.get("is-referenced-by-count", 0),
                        subjects=", ".join(str(x) for x in work.get("subject", []) or []),
                    )
                )
        except Exception as exc:
            if emit:
                emit("system", f"Crossref request failed for {query}: {type(exc).__name__}: {exc}")
    return sources, query_log


def retrieve_serpapi_scholar_sources(top_k_per_query: int, emit: YieldEmit | None = None) -> tuple[list[dict], list[dict]]:
    sources = []
    query_log = []
    if not Configs.SERPAPI_API_KEY:
        return sources, [{"source": "SerpAPI Google Scholar", "query": "", "error": "missing_serpapi_api_key"}]
    for query in _yield_scholar_queries():
        query_log.append({"source": "SerpAPI Google Scholar", "query": query})
        if emit:
            emit("search", f"Querying SerpAPI Google Scholar: {query}")
        try:
            resp = requests.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google_scholar",
                    "q": query,
                    "api_key": Configs.SERPAPI_API_KEY,
                    "num": top_k_per_query,
                },
                timeout=25,
            )
            if resp.status_code != 200:
                if emit:
                    emit("system", f"SerpAPI Google Scholar failed: HTTP {resp.status_code} for {query}")
                continue
            for item in resp.json().get("organic_results", []) or []:
                publication = item.get("publication_info") if isinstance(item.get("publication_info"), dict) else {}
                sources.append(
                    _as_source(
                        "serpapi_scholar",
                        "SerpAPI Google Scholar",
                        query,
                        item.get("title", ""),
                        item.get("link", ""),
                        item.get("snippet") or publication.get("summary", ""),
                        authors=publication.get("summary", ""),
                        cited_by_count=(item.get("inline_links") or {}).get("cited_by", {}).get("total", 0)
                        if isinstance(item.get("inline_links"), dict)
                        else 0,
                    )
                )
        except Exception as exc:
            if emit:
                emit("system", f"SerpAPI Google Scholar request failed for {query}: {type(exc).__name__}: {exc}")
    return sources, query_log


def retrieve_yield_sources(
    user_prompt: str,
    extra_queries: list[str] | None = None,
    top_k_per_query: int = 3,
    emit: YieldEmit | None = None,
) -> dict:
    plan = build_yield_query_plan(user_prompt, extra_queries)
    queries = plan[0]["queries"]

    all_sources: list[dict] = []
    query_log: list[dict] = []
    retrievers = [
        lambda: retrieve_web_sources(queries, top_k_per_query, emit),
        lambda: retrieve_arxiv_sources(top_k_per_query, emit),
        lambda: retrieve_semantic_scholar_sources(top_k_per_query, emit),
        lambda: retrieve_openalex_sources(top_k_per_query, emit),
        lambda: retrieve_crossref_sources(top_k_per_query, emit),
    ]
    if Configs.SERPAPI_API_KEY:
        retrievers.append(lambda: retrieve_serpapi_scholar_sources(top_k_per_query, emit))

    for fn in retrievers:
        sources, logs = fn()
        all_sources.extend(sources)
        query_log.extend(logs)

    numbered = _number_sources(_dedupe(all_sources, limit=40))
    return {
        "source_count": len(numbered),
        "provider_summary": _summary_by_key(numbered, "provider"),
        "source_type_summary": _summary_by_key(numbered, "source_type"),
        "source_quality": _quality_summary(numbered),
        "queries": query_log,
        "sources": numbered,
    }
