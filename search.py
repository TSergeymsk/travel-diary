"""
Поиск краткой информации о месте. Использует только Википедию —
без API-ключей, через публичный REST API.
"""
from __future__ import annotations

import httpx
from urllib.parse import quote

_cache: dict[tuple[str, str], str | None] = {}

_HEADERS = {"User-Agent": "travel-diary/1.0 (personal use)"}


def wiki_summary(
    place: str,
    lang: str = "ru",
    max_chars: int = 600,
    timeout: float = 8.0,
) -> str | None:
    if not place:
        return None
    key = (lang, place)
    if key in _cache:
        return _cache[key]

    try:
        # 1) Прямой заголовок
        r = httpx.get(
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(place)}",
            headers=_HEADERS, timeout=timeout, follow_redirects=True,
        )
        if r.status_code == 200:
            extract = r.json().get("extract")
            if extract:
                _cache[key] = extract[:max_chars]
                return _cache[key]

        # 2) Полнотекстовый поиск → первый результат
        r = httpx.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search",
                "srsearch": place, "format": "json", "srlimit": 1,
            },
            headers=_HEADERS, timeout=timeout,
        )
        hits = r.json().get("query", {}).get("search", [])
        if not hits:
            _cache[key] = None
            return None

        title = hits[0]["title"]
        r = httpx.get(
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
            headers=_HEADERS, timeout=timeout, follow_redirects=True,
        )
        if r.status_code == 200:
            extract = r.json().get("extract")
            if extract:
                _cache[key] = extract[:max_chars]
                return _cache[key]
    except Exception:
        pass

    _cache[key] = None
    return None