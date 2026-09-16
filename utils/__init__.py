import json as _json
import os
import requests
import sys
import time

from bs4 import BeautifulSoup
from urllib.parse import unquote
# SerpAPI removed — using Google Custom Search API (CSE) directly

from openai import OpenAI
from configs import AVAILABLE_LLMs, Configs, LLM_REQUEST_TIMEOUT, get_llm_model_fallbacks

# ---------------------------------------------------------------------------
# Live event log — written as JSONL for real-time dashboard consumption
# ---------------------------------------------------------------------------
_EVENT_LOG_PATH = os.environ.get("AMLA_EVENT_LOG", "")
_event_start_time = time.time()
_SEARCHAPI_DISABLED_FOR_RUN = False


def _emit_event(sender, label, msg, mirror=None):
    """Append a structured event to the JSONL log file (if enabled)."""
    if mirror is None:
        mirror = os.environ.get("AMLA_MIRROR_EVENTS_TO_STDOUT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if mirror:
        try:
            print(f"{label} {msg}", flush=True)
        except UnicodeEncodeError:
            safe = f"{label} {msg}".encode("ascii", errors="replace").decode("ascii")
            print(safe, flush=True)
    if not _EVENT_LOG_PATH:
        return
    event = {
        "ts": time.time(),
        "elapsed": round(time.time() - _event_start_time, 2),
        "sender": sender,
        "label": label,
        "content": str(msg)[:20000],
    }
    try:
        with open(_EVENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def get_kaggle():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


# def search_web(query):
#     try:
#         # Abort the request after 10 seconds
#         response = requests.get(f"https://www.google.com/search?hl=en&q={query}")
#         response.raise_for_status()  # Raises an HTTPError for bad responses
#         html_string = response.text
#     except requests.exceptions.RequestException as e:
#         print_message(
#             "system",
#             "Request Google Search Failed with " + str(e) + "\n Using SerpAPI.",
#         )
#         params = {
#             "engine": "google",
#             "q": query,
#             "api_key": "",
#         }

#         search = GoogleSearch(params)
#         results = search.get_dict()
#         return results["organic_results"]

#     # Parse the HTML content
#     soup = BeautifulSoup(html_string, "html.parser")

#     # Find all <a> tags
#     links = soup.find_all("a")

#     if not links:
#         raise Exception('Webpage does not have any "a" element')

#     # Filter and process the links
#     filtered_links = []
#     for link in links:
#         href = link.get("href")
#         if href and href.startswith("/url?q=") and "google.com" not in href:
#             cleaned_link = unquote(
#                 href.split("&sa=")[0][7:]
#             )  # Remove "/url?q=" and split at "&sa="
#             filtered_links.append(cleaned_link)

#     # Remove duplicates and prepare the output
#     unique_links = list(set(filtered_links))
#     return {"organic_results": [{"link": link} for link in unique_links]}[
#         "organic_results"
#     ]

def _search_google_cse(query):
    # Prefer the explicit Google key. For backward compatibility, accept the
    # old SEARCHAPI_API_KEY only when it looks like a Google API key.
    api_key = Configs.GOOGLE_API_KEY or (
        Configs.SEARCHAPI_API_KEY if Configs.SEARCHAPI_API_KEY.startswith("AIza") else ""
    )
    cse_id  = Configs.GOOGLE_CSE_ID
    if not api_key or not cse_id:
        try:
            print_message("system", "Google CSE is not configured: missing GOOGLE_API_KEY/Google-style SEARCHAPI_API_KEY or GOOGLE_CSE_ID.")
        except Exception:
            pass
        return []

    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cse_id, "q": query, "num": 10},
            timeout=10,
        )
        if resp.status_code != 200:
            try:
                detail = resp.text[:500].replace("\n", " ")
                print_message("system", f"Google CSE request failed: HTTP {resp.status_code}. {detail}")
            except Exception:
                pass
            return []
        items = resp.json().get("items", [])
        # Normalise to the same shape the rest of the code expects
        return [{"title": it.get("title", ""), "link": it.get("link", ""),
                 "snippet": it.get("snippet", ""), "provider": "Google CSE"} for it in items]
    except Exception as e:
        try:
            print_message("system", f"Google CSE request failed: {type(e).__name__}. Check network/DNS access to www.googleapis.com.")
        except Exception:
            pass
        return []


def _search_searchapi(query):
    global _SEARCHAPI_DISABLED_FOR_RUN
    if _SEARCHAPI_DISABLED_FOR_RUN:
        return []
    api_key = Configs.SEARCHAPI_API_KEY
    if not api_key or api_key.startswith("AIza"):
        try:
            print_message("system", "SearchAPI.io is not configured: missing SEARCHAPI_API_KEY, or SEARCHAPI_API_KEY still contains a Google Cloud key.")
        except Exception:
            pass
        return []

    last_error = None
    for attempt in range(1, 3):
        try:
            resp = requests.get(
                "https://www.searchapi.io/api/v1/search",
                params={
                    "engine": "google",
                    "q": query,
                    "api_key": api_key,
                    "num": 10,
                },
                timeout=35,
            )
            if resp.status_code != 200:
                detail = resp.text[:500].replace("\n", " ")
                print_message("system", f"SearchAPI.io request failed: HTTP {resp.status_code}. {detail}")
                if resp.status_code == 429 or "used all of the searches" in detail.lower() or "quota" in detail.lower():
                    _SEARCHAPI_DISABLED_FOR_RUN = True
                    print_message("manager", "SearchAPI.io quota/rate limit detected; disabling SearchAPI.io for the rest of this run.")
                return []
            payload = resp.json()
            items = payload.get("organic_results") or payload.get("results") or []
            normalised = []
            for item in items:
                normalised.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", item.get("url", "")),
                    "snippet": item.get("snippet", item.get("description", "")),
                    "provider": "SearchAPI.io",
                })
            return normalised
        except requests.exceptions.ReadTimeout as e:
            last_error = e
            print_message("system", f"SearchAPI.io request timed out on attempt {attempt}/2; retrying." if attempt == 1 else "SearchAPI.io request timed out on attempt 2/2.")
        except Exception as e:
            last_error = e
            break
    try:
        print_message("system", f"SearchAPI.io request failed: {type(last_error).__name__}. Check network/DNS access to www.searchapi.io.")
    except Exception:
        pass
    return []


def _search_tavily(query):
    api_key = Configs.TAVILY_API_KEY
    if not api_key:
        try:
            print_message("system", "Tavily is not configured: missing TAVILY_API_KEY.")
        except Exception:
            pass
        return []

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": 10,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            try:
                detail = resp.text[:500].replace("\n", " ")
                print_message("system", f"Tavily request failed: HTTP {resp.status_code}. {detail}")
            except Exception:
                pass
            return []
        items = resp.json().get("results", [])
        return [
            {
                "title": it.get("title", ""),
                "link": it.get("url", ""),
                "snippet": it.get("content", ""),
                "provider": "Tavily",
            }
            for it in items
        ]
    except Exception as e:
        try:
            print_message("system", f"Tavily request failed: {type(e).__name__}. Check network/DNS access to api.tavily.com.")
        except Exception:
            pass
    return []


def _search_brave(query):
    api_key = Configs.BRAVE_SEARCH_API_KEY
    if not api_key:
        try:
            print_message("system", "Brave Search is not configured: missing BRAVE_SEARCH_API_KEY.")
        except Exception:
            pass
        return []

    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
            },
            params={
                "q": query,
                "count": 10,
                "safesearch": "moderate",
                "extra_snippets": "true",
            },
            timeout=20,
        )
        if resp.status_code != 200:
            try:
                detail = resp.text[:500].replace("\n", " ")
                print_message("system", f"Brave Search request failed: HTTP {resp.status_code}. {detail}")
            except Exception:
                pass
            return []
        items = ((resp.json().get("web") or {}).get("results") or [])
        normalised = []
        for item in items:
            snippets = item.get("extra_snippets") or []
            snippet = item.get("description", "")
            if snippets:
                snippet = " ".join([snippet] + [str(x) for x in snippets if x])
            normalised.append({
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "snippet": snippet,
                "provider": "Brave Search",
            })
        return normalised
    except Exception as e:
        try:
            print_message("system", f"Brave Search request failed: {type(e).__name__}. Check network/DNS access to api.search.brave.com.")
        except Exception:
            pass
        return []


def _search_exa(query):
    api_key = Configs.EXA_API_KEY
    if not api_key:
        try:
            print_message("system", "Exa is not configured: missing EXA_API_KEY.")
        except Exception:
            pass
        return []

    try:
        resp = requests.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "numResults": 10,
                "contents": {"text": True, "highlights": True},
            },
            timeout=25,
        )
        if resp.status_code != 200:
            try:
                detail = resp.text[:500].replace("\n", " ")
                print_message("system", f"Exa request failed: HTTP {resp.status_code}. {detail}")
            except Exception:
                pass
            return []
        normalised = []
        for item in resp.json().get("results", []) or []:
            highlights = item.get("highlights") or []
            snippet = " ".join(str(x) for x in highlights if x) or item.get("text", "") or item.get("summary", "")
            normalised.append({
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "snippet": snippet,
                "provider": "Exa",
            })
        return normalised
    except Exception as e:
        try:
            print_message("system", f"Exa request failed: {type(e).__name__}. Check network/DNS access to api.exa.ai.")
        except Exception:
            pass
        return []


def _search_serpapi(query):
    api_key = Configs.SERPAPI_API_KEY
    if not api_key:
        try:
            print_message("system", "SerpAPI is not configured: missing SERPAPI_API_KEY.")
        except Exception:
            pass
        return []

    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": 10,
            },
            timeout=25,
        )
        if resp.status_code != 200:
            try:
                detail = resp.text[:500].replace("\n", " ")
                print_message("system", f"SerpAPI request failed: HTTP {resp.status_code}. {detail}")
            except Exception:
                pass
            return []
        normalised = []
        for item in resp.json().get("organic_results", []) or []:
            normalised.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "provider": "SerpAPI Google",
            })
        return normalised
    except Exception as e:
        try:
            print_message("system", f"SerpAPI request failed: {type(e).__name__}. Check network/DNS access to serpapi.com.")
        except Exception:
            pass
        return []


def search_web(query):
    """Search the web through the configured provider.

    SEARCH_PROVIDER:
    - auto: try configured external providers until one returns results
    - google: Google CSE only
    - searchapi: SearchAPI.io only
    - tavily: Tavily only
    - brave: Brave Search only
    - exa: Exa only
    - serpapi: SerpAPI Google Search only
    """
    provider = Configs.SEARCH_PROVIDER or "auto"
    if provider == "searchapi":
        return _search_searchapi(query)
    if provider == "tavily":
        return _search_tavily(query)
    if provider == "google":
        return _search_google_cse(query)
    if provider == "brave":
        return _search_brave(query)
    if provider == "exa":
        return _search_exa(query)
    if provider == "serpapi":
        return _search_serpapi(query)

    if Configs.SEARCHAPI_API_KEY and not Configs.SEARCHAPI_API_KEY.startswith("AIza"):
        results = _search_searchapi(query)
        if results:
            return results
    google_api_key = Configs.GOOGLE_API_KEY or (
        Configs.SEARCHAPI_API_KEY if Configs.SEARCHAPI_API_KEY.startswith("AIza") else ""
    )
    if google_api_key and Configs.GOOGLE_CSE_ID:
        results = _search_google_cse(query)
        if results:
            return results
    if Configs.TAVILY_API_KEY:
        print_message("manager", "Trying Tavily external provider.")
        results = _search_tavily(query)
        if results:
            return results
    if Configs.BRAVE_SEARCH_API_KEY:
        print_message("manager", "Trying Brave Search external provider.")
        results = _search_brave(query)
        if results:
            return results
    if Configs.EXA_API_KEY:
        print_message("manager", "Trying Exa external provider.")
        results = _search_exa(query)
        if results:
            return results
    if Configs.SERPAPI_API_KEY:
        print_message("manager", "Trying SerpAPI external provider.")
        results = _search_serpapi(query)
        if results:
            return results
    return []


def print_message(sender, msg, pid=None):
    pid = f"-{pid}" if pid else ""
    sender_color = {
        "user": color.PURPLE,
        "system": color.RED,
        "manager": color.GREEN,
        "model": color.BLUE,
        "data": color.DARKCYAN,
        "prompt": color.CYAN,
        "operation": color.YELLOW,
    }
    sender_label = {
        "user": "You:",
        "system": "SYSTEM NOTICE:\n",
        "manager": "Agent Manager:",
        "model": f"Model Agent{pid}:",
        "data": f"Data Agent{pid}:",
        "prompt": "Prompt Agent:",
        "operation": f"Operation Agent{pid}:",
    }

    rendered = f"{color.BOLD}{sender_color[sender]}{sender_label[sender]}{color.END}{color.END} {msg}"
    try:
        print(rendered, flush=True)
    except UnicodeEncodeError:
        safe = rendered.encode("ascii", errors="replace").decode("ascii")
        print(safe, flush=True)
    print(flush=True)

    # Emit to live dashboard log
    _emit_event(sender, sender_label.get(sender, sender), msg, mirror=False)


def _is_fallbackable_llm_error(exc) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "quota",
            "free tier",
            "allocationquota",
            "permissiondenied",
            "permission denied",
            "insufficient_quota",
            "model_not_found",
            "model not found",
            "model_not_available",
            "not available",
            "access denied",
            "403",
            "429",
            "timeout",
            "timed out",
            "request timed out",
        )
    )


class _FallbackChatCompletions:
    def __init__(self, raw_client, llm: str):
        self._raw_client = raw_client
        self._llm = llm

    def create(self, *args, **kwargs):
        primary_model = kwargs.get("model")
        models = get_llm_model_fallbacks(self._llm, primary_model)
        if not models:
            return self._raw_client.chat.completions.create(*args, **kwargs)

        last_error = None
        for index, model in enumerate(models):
            call_kwargs = dict(kwargs)
            call_kwargs["model"] = model
            try:
                if index > 0:
                    _emit_event(
                        "system",
                        "SYSTEM",
                        f"LLM fallback switched to model={model}",
                        mirror=False,
                    )
                return self._raw_client.chat.completions.create(*args, **call_kwargs)
            except Exception as exc:
                last_error = exc
                _emit_event(
                    "system",
                    "SYSTEM",
                    f"LLM model failed: model={model}; error={type(exc).__name__}: {exc}",
                    mirror=False,
                )
                if not _is_fallbackable_llm_error(exc) or index == len(models) - 1:
                    raise
        raise last_error


class _FallbackChat:
    def __init__(self, raw_client, llm: str):
        self.completions = _FallbackChatCompletions(raw_client, llm)


class _FallbackOpenAIClient:
    def __init__(self, raw_client, llm: str):
        self._raw_client = raw_client
        self.chat = _FallbackChat(raw_client, llm)

    def __getattr__(self, name):
        return getattr(self._raw_client, name)


def get_client(llm: str = "qwen"):
    client_config = AVAILABLE_LLMs[llm]
    client_kwargs = {"api_key": client_config["api_key"], "timeout": LLM_REQUEST_TIMEOUT}
    if "base_url" in client_config:
        client_kwargs["base_url"] = client_config["base_url"]
    raw_client = OpenAI(**client_kwargs)
    if client_config.get("model_fallbacks"):
        return _FallbackOpenAIClient(raw_client, llm)
    return raw_client
