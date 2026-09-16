import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)
except ImportError:
    pass


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_EXHAUSTED_ARK_MODELS = {
    "qwen-plus",
    "qwen3.7-max-2026-05-17",
    "qwen3.7-max-preview",
    "qwen3.7-plus-2026-05-26",
    "kimi-k2.6",
    "deepseek-v4-pro",
    "qwen3.6-27b",
    "qwen3.6-flash",
}
_DEFAULT_ARK_MODEL = "qwen3.7-flash-2026-07-15"
_DEFAULT_ARK_FALLBACKS = [
    "qwen3.7-flash",
]


def _usable_ark_model(value: str | None) -> str:
    model = str(value or "").strip()
    if not model or model in _EXHAUSTED_ARK_MODELS:
        return _DEFAULT_ARK_MODEL
    return model


def _usable_ark_fallbacks(value: str | None) -> str:
    requested = [item.strip() for item in str(value or "").split(",") if item.strip()]
    models = requested or list(_DEFAULT_ARK_FALLBACKS)
    cleaned = []
    seen = {_DEFAULT_ARK_MODEL}
    for model in models:
        if model in _EXHAUSTED_ARK_MODELS or model in seen:
            continue
        seen.add(model)
        cleaned.append(model)
    if not cleaned:
        cleaned = list(_DEFAULT_ARK_FALLBACKS)
    return ",".join(cleaned)


class Configs:
    OPENAI_KEY = os.getenv("OPENAI_KEY", "")  # your openai's account api key
    CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY", "")
    CUSTOM_BASE_URL = os.getenv("CUSTOM_BASE_URL", "https://z.apiyihe.org/v1")
    CUSTOM_MODEL = os.getenv("CUSTOM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    HF_KEY = os.getenv("HF_KEY", "")
    PWC_KEY = os.getenv("PWC_KEY", "")
    # SearchAPI.io key. Older local configs may have stored a Google Cloud key
    # here; Google CSE now also accepts GOOGLE_API_KEY explicitly.
    SEARCHAPI_API_KEY = os.getenv("SEARCHAPI_API_KEY", "")
    GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CSE_ID     = os.getenv("GOOGLE_CSE_ID", "")      # Programmable Search Engine ID
    TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")
    BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
    EXA_API_KEY       = os.getenv("EXA_API_KEY", "")
    SERPAPI_API_KEY   = os.getenv("SERPAPI_API_KEY", "")
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    CROSSREF_MAILTO   = os.getenv("CROSSREF_MAILTO", "")
    OPENALEX_EMAIL    = os.getenv("OPENALEX_EMAIL", "")     # optional, used for OpenAlex polite pool
    OPENALEX_API_KEY  = os.getenv("OPENALEX_API_KEY", "")   # optional, if using an OpenAlex account key
    SEARCH_PROVIDER   = os.getenv("SEARCH_PROVIDER", "auto").strip().lower()  # auto/google/searchapi/tavily/brave/exa/serpapi
    SEARCH_WEB_ENABLED = _as_bool(os.getenv("SEARCH_WEB_ENABLED"), default=True)
    SEARCH_FETCH_PAGES = _as_bool(os.getenv("SEARCH_FETCH_PAGES"), default=False)

    ARK_API_KEY = os.getenv("ARK_API_KEY", "")
    ARK_BASE_URL = os.getenv(
        "ARK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    # Default to models that currently have free quota in the user's account.
    # Keep coding-capable models before smaller fallbacks, and avoid exhausted
    # max/free-tier endpoints in the default chain.
    ARK_MODEL = _usable_ark_model(os.getenv("ARK_MODEL", _DEFAULT_ARK_MODEL))
    ARK_MODEL_FALLBACKS = _usable_ark_fallbacks(os.getenv("ARK_MODEL_FALLBACKS", ""))

    RAG_ENABLED = _as_bool(os.getenv("RAG_ENABLED"), default=False)
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
    RAG_BM25_TOP_K = int(os.getenv("RAG_BM25_TOP_K", "8"))
    RAG_VECTOR_TOP_K = int(os.getenv("RAG_VECTOR_TOP_K", "8"))
    RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "./data/rag_index")

    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "thenlper/gte-small")
    ARK_EMBEDDING_API_KEY = os.getenv("ARK_EMBEDDING_API_KEY", "")
    ARK_EMBEDDING_BASE_URL = os.getenv(
        "ARK_EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    ARK_EMBEDDING_MODEL = os.getenv("ARK_EMBEDDING_MODEL", "text-embedding-v3")


USE_OPENAI_PARSER = _as_bool(os.getenv("USE_OPENAI_PARSER"), default=True)
PARSER_LLM = os.getenv("PARSER_LLM", "ark")
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "ark")
LOW_TOKEN_MODE = _as_bool(os.getenv("LOW_TOKEN_MODE"), default=False)
LOW_TOKEN_N_PLANS = int(os.getenv("LOW_TOKEN_N_PLANS", "1"))
LOW_TOKEN_N_CANDIDATES = int(os.getenv("LOW_TOKEN_N_CANDIDATES", "1"))
LOW_TOKEN_N_REVISE = int(os.getenv("LOW_TOKEN_N_REVISE", "0"))
OPERATION_MAX_ERROR_CHARS = int(os.getenv("OPERATION_MAX_ERROR_CHARS", "4000"))
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "240"))


AVAILABLE_LLMs = {  
    "prompt-llm": {
        "api_key": "empty",
        "model": "prompt-llama",
        "base_url": "http://localhost:8000/v1",
    },
    "gpt-4.1": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4.1"},
    "gpt-4": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4o"},
    "gpt-3.5": {"api_key": Configs.OPENAI_KEY, "model": "gpt-3.5-turbo"},
    "custom": {
        "api_key": Configs.CUSTOM_API_KEY,
        "model": Configs.CUSTOM_MODEL,
        "base_url": Configs.CUSTOM_BASE_URL,
    },
    "ark": {
        "api_key": Configs.ARK_API_KEY,
        "model": Configs.ARK_MODEL,
        "base_url": Configs.ARK_BASE_URL,
        "model_fallbacks": Configs.ARK_MODEL_FALLBACKS,
    },
}


def _split_model_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def get_llm_model_fallbacks(llm: str, primary_model: str | None = None) -> list[str]:
    """Return provider-compatible model candidates in priority order."""
    cfg = AVAILABLE_LLMs.get(llm, {})
    candidates = []
    if primary_model:
        candidates.append(primary_model)
    elif cfg.get("model"):
        candidates.append(str(cfg["model"]))
    candidates.extend(_split_model_list(str(cfg.get("model_fallbacks", ""))))

    seen = set()
    ordered = []
    for model in candidates:
        if model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered

DOMAIN_KNOWLEDGE_ENABLED = _as_bool(os.getenv("DOMAIN_KNOWLEDGE_ENABLED"), default=False)

TASK_METRICS = {
    "image_classification": "accuracy",
    "text_classification": "accuracy",
    "tabular_classification": "F1",
    "tabular_regression": "RMSLE",
    "tabular_clustering": "RI",
    "node_classification": "accuracy",
    "ts_forecasting": "RMSLE",
    "yield_stress_regression": "R2",
}

# The yield-stress pipeline is orchestrated by run_yield.py. The legacy
# AgentManager domain-task path is intentionally disabled.
DOMAIN_TASKS = set()
