"""
Поиск краткой информации о месте через Википедию.
Без API-ключей, через публичный REST API.
"""
from __future__ import annotations

import httpx
from urllib.parse import quote

_cache: dict[tuple[str, str], str | None] = {}

# Wikipedia требует контактный URL или email в User-Agent.
# Замените на свой реальный контакт — это их политика.
_HEADERS = {
    "User-Agent": "TravelDiary/1.0 (https://example.com/travel-diary; mailto:you@example.com)",
    "Accept": "application/json",
}


def _fetch(url: str, timeout: float, params: dict | None = None) -> dict | None:
    try:
        r = httpx.get(url, headers=_HEADERS, params=params,
                      timeout=timeout, follow_redirects=True)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def wiki_summary(place: str, lang: str = "ru", max_chars: int = 600,
                 timeout: float = 8.0) -> str | None:
    if not place:
        return None
    key = (lang, place)
    if key in _cache:
        return _cache[key]

    # 1) Прямой заголовок через REST summary
    data = _fetch(
        f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(place)}",
        timeout,
    )
    if data and data.get("extract"):
        _cache[key] = data["extract"][:max_chars]
        return _cache[key]

    # 2) Полнотекстовый поиск → extract через action API
    data = _fetch(
        f"https://{lang}.wikipedia.org/w/api.php",
        timeout,
        params={
            "action": "query",
            "list": "search",
            "srsearch": place,
            "format": "json",
            "srlimit": 1,
        },
    )
    hits = (data or {}).get("query", {}).get("search", [])
    if not hits:
        _cache[key] = None
        return None

    title = hits[0]["title"]
    data = _fetch(
        f"https://{lang}.wikipedia.org/w/api.php",
        timeout,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "redirects": 1,
        },
    )
    pages = (data or {}).get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract")
        if extract:
            _cache[key] = extract[:max_chars]
            return _cache[key]

    _cache[key] = None
    return None