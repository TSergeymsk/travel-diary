"""
Поиск краткой информации о месте через Википедию.
Без API-ключей, через публичный REST API.

Основной путь — геопоиск по координатам (list=geosearch). Это позволяет
отличить, например, район Чаоян в Пекине от города Чаоян в Ляонине.
Текстовый поиск остаётся как резерв, когда координат нет.
"""
from __future__ import annotations

import httpx
from urllib.parse import quote

_cache: dict = {}

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
    # сохраняем порядок titles: сначала тот, что первым в списке
    order = {}
    for i, t in enumerate(titles):
        order[t.lower()] = i
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
    if best[1]:
        return best[1][:max_chars]
    return None


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
    """Текстовый поиск по имени, возвращает список заголовков."""
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
    затем по расстоянию. Возвращает упорядоченный список заголовков.
    """
    toponym = ""
    if place:
        toponym = place.split(",")[0].strip().lower()

    def key(item: tuple[str, float]):
        title, dist = item
        t = title.lower()
        if toponym:
            # «район Чаоян» ~ «Чаоян» ~ «Район Чаоян»
            match = toponym in t or t in toponym
            prefix_bonus = 0 if t.startswith(toponym) else 1
        else:
            match = False
            prefix_bonus = 0
        # 0 — совпало, 1 — нет; внутри группы — по расстоянию
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
) -> str | None:
    """
    Ищет краткую справку о месте.
    Приоритет:
      1. Геопоиск по координатам (если lat/lon заданы) — самый точный.
      2. Текстовый поиск по названию.
    """
    if lat is None and lon is None and not place:
        return None

    ck = (lang, place or "",
          round(lat, 3) if lat is not None else None,
          round(lon, 3) if lon is not None else None)
    if ck in _cache:
        return _cache[ck]

    result: str | None = None

    # ---- 1. Геопоиск ----
    if lat is not None and lon is not None:
        hits = _geosearch(lat, lon, lang, timeout, radius_m)
        if hits:
            titles = _rank_by_toponym(hits, place)
            result = _extract_for_titles(titles, lang, timeout, max_chars)
            if result:
                _cache[ck] = result
                return result

    # ---- 2. Текстовый поиск ----
    if place:
        # Сначала пробуем прямой заголовок через REST summary
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