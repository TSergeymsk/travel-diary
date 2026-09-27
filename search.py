"""
Поиск краткой информации о месте через Википедию.
Без API-ключей, через публичный REST API.

Стратегия:
  1. Геопоиск по координатам (list=geosearch).
     Если ближайшая статья дальше max_dist_m — считаем её «не про это место»
     и переходим к текстовому поиску. Это спасает от того, что рядом с
     Химками статью про безымянную деревню в 137 человек геопоиск находит
     раньше, чем собственно «Химки».
  2. Текстовый поиск по названию (list=search + REST summary).
"""
from __future__ import annotations

import httpx
from urllib.parse import quote

_cache: dict = {}

# Wikipedia требует контактный URL или email в User-Agent.
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


def _extract_for_titles(
    titles: list[str], lang: str, timeout: float, max_chars: int
) -> str | None:
    """Забирает intro-секцию для первого заголовка из списка, у которого она есть."""
    if not titles:
        return None
    data = _fetch(
        f"https://{lang}.wikipedia.org/w/api.php",
        timeout,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "titles": "|".join(titles[:5]),
            "format": "json",
            "redirects": 1,
        },
    )
    pages = (data or {}).get("query", {}).get("pages", {})
    order = {t.lower(): i for i, t in enumerate(titles)}
    best: tuple[int, str] = (10_000, "")
    for page in pages.values():
        if page.get("missing"):
            continue
        ext = page.get("extract")
        if not ext:
            continue
        title = page.get("title", "").lower()
        rank = order.get(title, 1000)
        if rank < best[0]:
            best = (rank, ext)
    return best[1][:max_chars] if best[1] else None


def _geosearch(
    lat: float, lon: float, lang: str, timeout: float, radius_m: int
) -> list[tuple[str, float]]:
    """Возвращает список (title, distance_m) из окрестности точки."""
    data = _fetch(
        f"https://{lang}.wikipedia.org/w/api.php",
        timeout,
        params={
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": radius_m,
            "gslimit": 10,
            "format": "json",
        },
    )
    hits = (data or {}).get("query", {}).get("geosearch", []) or []
    return [(h["title"], h.get("dist", 99999)) for h in hits]


def _text_search(place: str, lang: str, timeout: float) -> list[str]:
    data = _fetch(
        f"https://{lang}.wikipedia.org/w/api.php",
        timeout,
        params={
            "action": "query",
            "list": "search",
            "srsearch": place,
            "format": "json",
            "srlimit": 5,
        },
    )
    return [h["title"] for h in (data or {}).get("query", {}).get("search", [])]


def _rank_by_toponym(
    hits: list[tuple[str, float]], place: str | None
) -> list[str]:
    """
    Сортирует геохиты: сначала те, что по названию совпадают с топонимом,
    затем по расстоянию.
    """
    toponym = ""
    if place:
        toponym = place.split(",")[0].strip().lower()

    def key(item: tuple[str, float]):
        title, dist = item
        t = title.lower()
        if toponym:
            match = toponym in t or t in toponym
            prefix_bonus = 0 if t.startswith(toponym) else 1
        else:
            match = False
            prefix_bonus = 0
        return (0 if match else 1, prefix_bonus, dist)

    return [title for title, _ in sorted(hits, key=key)]


def wiki_summary(
    place: str | None,
    lat: float | None = None,
    lon: float | None = None,
    lang: str = "ru",
    max_chars: int = 600,
    timeout: float = 8.0,
    radius_m: int = 5000,
    max_dist_m: int = 2000,
) -> str | None:
    """
    Ищет краткую справку о месте.

    max_dist_m — максимальное расстояние от точки до статьи геопоиска,
    при котором считаем, что статья — «про это место». Если ближайшая
    статья дальше, уходим в текстовый поиск по имени.
    """
    if lat is None and lon is None and not place:
        return None

    ck = (lang, place or "",
          round(lat, 3) if lat is not None else None,
          round(lon, 3) if lon is not None else None,
          max_dist_m)
    if ck in _cache:
        return _cache[ck]

    result: str | None = None

    # ---- 1. Геопоиск (только если точка достаточно близко) ----
    if lat is not None and lon is not None:
        hits = _geosearch(lat, lon, lang, timeout, radius_m)
        close_hits = [(t, d) for t, d in hits if d <= max_dist_m]
        if close_hits:
            titles = _rank_by_toponym(close_hits, place)
            result = _extract_for_titles(titles, lang, timeout, max_chars)
            if result:
                _cache[ck] = result
                return result

    # ---- 2. Текстовый поиск ----
    if place:
        data = _fetch(
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
            f"{quote(place)}",
            timeout,
        )
        if data and data.get("extract"):
            result = data["extract"][:max_chars]
            _cache[ck] = result
            return result

        titles = _text_search(place, lang, timeout)
        if titles:
            result = _extract_for_titles(titles, lang, timeout, max_chars)
            if result:
                _cache[ck] = result
                return result

    _cache[ck] = None
    return None