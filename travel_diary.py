#!/usr/bin/env python3
"""
travel_diary.py — генерирует HTML-дневник путешествия.

Пайплайн:
  1. Метаданные   — обход всех файлов, EXIF + mtime-fallback + tz-offset
  2. GPX          — расстояния по дням + интерполяция GPS
  3. Кластеры     — группировка фото по дням и локациям
  4. Геокодинг    — Nominatim + Википедия (геопоиск по координатам)
  5. Фаза VISION  — описания кадров для каждого кластера
  6. Классификация — major/minor локации
  7. Фаза TEXT    — заголовки, рассказы по локациям, вступления дней,
                    обзор маршрута, финал
  8. Фаза PROOFREAD — вычитка
  9. Экспорт фото + HTML (с дедупликацией и диверсификацией кадров)
"""
from __future__ import annotations

import argparse
import bisect
import io
import logging
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import exifread
import gpxpy
import httpx
import yaml
from jinja2 import Environment, FileSystemLoader
from PIL import Image, ImageOps

from providers import Registry
from search import wiki_summary


# ============================================================
#  Утилиты
# ============================================================

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def dms_to_deg(values, ref) -> float:
    d, m, s = (float(v.num) / float(v.den) for v in values)
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def parse_tz_offset(s: str) -> timezone | None:
    if not s:
        return None
    s = s.strip()
    try:
        sign = 1 if s[0] == "+" else -1
        hh, mm = s[1:].split(":")
        return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))
    except Exception:
        return None


def dedup_consecutive(items: list[str]) -> list[str]:
    """Убирает подряд идущие одинаковые строки."""
    out: list[str] = []
    for x in items:
        if not out or out[-1] != x:
            out.append(x)
    return out


# ============================================================
#  Модели данных
# ============================================================

@dataclass
class Photo:
    path: Path
    dt: datetime | None = None
    lat: float | None = None
    lon: float | None = None
    camera: str | None = None
    gps_source: str | None = None
    dt_source: str | None = None
    tz: timezone | None = None
    _phash: int | None = field(default=None, repr=False)

    @property
    def day(self):
        return self.dt.date() if self.dt else None

    def phash(self) -> int | None:
        if self._phash is None:
            try:
                img = Image.open(self.path)
                img = ImageOps.exif_transpose(img)
                self._phash = _phash_value(img)
            except Exception:
                return None
        return self._phash


@dataclass
class Cluster:
    photos: list[Photo] = field(default_factory=list)
    lat: float | None = None
    lon: float | None = None
    place: str | None = None
    wiki: str | None = None
    title: str = ""
    description: str = ""
    narrative: str = ""
    significant: bool = False

    @property
    def time_start(self) -> datetime | None:
        return min((p.dt for p in self.photos if p.dt), default=None)

    @property
    def time_end(self) -> datetime | None:
        return max((p.dt for p in self.photos if p.dt), default=None)

    def recompute(self):
        pts = [(p.lat, p.lon) for p in self.photos if p.lat is not None]
        if pts:
            self.lat = sum(p[0] for p in pts) / len(pts)
            self.lon = sum(p[1] for p in pts) / len(pts)


@dataclass
class Day:
    date: object
    photos: list[Photo] = field(default_factory=list)
    clusters: list[Cluster] = field(default_factory=list)
    distance_km: float = 0.0
    title: str = ""
    places_summary: str = ""
    day_intro: str = ""
    major_clusters: list[Cluster] = field(default_factory=list)
    minor_clusters: list[Cluster] = field(default_factory=list)


# ============================================================
#  Этап 1: сбор метаданных
# ============================================================

EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff"}


def read_photo(path: Path, use_mtime_fallback: bool) -> Photo | None:
    p = Photo(path=path)
    try:
        with path.open("rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return None

    for key in ("EXIF DateTimeOriginal", "Image DateTime", "EXIF DateTimeDigitized"):
        if key in tags:
            try:
                p.dt = datetime.strptime(str(tags[key]), "%Y:%m:%d %H:%M:%S")
                p.dt_source = "exif"
                break
            except ValueError:
                pass

    if p.dt is None and use_mtime_fallback:
        try:
            p.dt = datetime.fromtimestamp(path.stat().st_mtime)
            p.dt_source = "mtime"
        except OSError:
            pass

    if p.dt is None:
        return None

    for key in ("EXIF OffsetTimeOriginal", "EXIF OffsetTime",
                "EXIF OffsetTimeDigitized"):
        if key in tags:
            p.tz = parse_tz_offset(str(tags[key]))
            if p.tz:
                break
    if p.tz is not None and p.dt.tzinfo is None:
        p.dt = p.dt.replace(tzinfo=p.tz)

    try:
        p.lat = dms_to_deg(tags["GPS GPSLatitude"].values,
                           tags["GPS GPSLatitudeRef"].values)
        p.lon = dms_to_deg(tags["GPS GPSLongitude"].values,
                           tags["GPS GPSLongitudeRef"].values)
        p.gps_source = "exif"
    except (KeyError, AttributeError, ZeroDivisionError):
        pass

    if "Image Model" in tags:
        p.camera = str(tags["Image Model"]).strip()
    return p


def collect_metadata(
    root: Path,
    use_mtime_fallback: bool,
    fallback_tz: timezone,
) -> tuple[list[Photo], dict]:
    """Обходит всю папку, читает метаданные. Все даты — aware."""
    all_files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTS:
            all_files.append(path)

    photos: list[Photo] = []
    stats = {
        "files": len(all_files),
        "no_dt": 0,
        "no_gps": 0,
        "no_tz": 0,
        "from_mtime": 0,
        "broken": 0,
    }

    for path in all_files:
        try:
            p = read_photo(path, use_mtime_fallback)
        except Exception:
            stats["broken"] += 1
            continue
        if p is None:
            stats["no_dt"] += 1
            continue

        if p.dt is not None and p.dt.tzinfo is None:
            if p.tz is not None:
                p.dt = p.dt.replace(tzinfo=p.tz)
            else:
                p.dt = p.dt.replace(tzinfo=fallback_tz)
                stats["no_tz"] += 1

        photos.append(p)
        if p.dt_source == "mtime":
            stats["from_mtime"] += 1
        if p.lat is None:
            stats["no_gps"] += 1

    photos.sort(key=lambda x: x.dt)
    return photos, stats


# ============================================================
#  Этап 2: GPX
# ============================================================

def load_gpx(path: Path):
    with path.open() as f:
        return gpxpy.parse(f)


def _collect_track_points_by_day(gpx, photo_tz: timezone) -> dict:
    by_day: dict = defaultdict(list)
    for track in gpx.tracks:
        for seg in track.segments:
            for pt in seg.points:
                if not pt.time:
                    continue
                t = pt.time
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                local_date = t.astimezone(photo_tz).date()
                by_day[local_date].append((t, pt.latitude, pt.longitude))
    for d in by_day:
        by_day[d].sort(key=lambda x: x[0])
    return by_day


def track_distance_for_day(gpx, day_date, photo_tz: timezone) -> float:
    total = 0.0
    prev = None
    for track in gpx.tracks:
        for seg in track.segments:
            for pt in seg.points:
                if not pt.time:
                    prev = None
                    continue
                t = pt.time
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if t.astimezone(photo_tz).date() != day_date:
                    prev = None
                    continue
                if prev is not None:
                    total += haversine(prev[0], prev[1], pt.latitude, pt.longitude)
                prev = (pt.latitude, pt.longitude)
    return total / 1000.0


def interpolate_gps_from_gpx(
    photos: list[Photo],
    gpx,
    max_gap_min: int = 30,
    fallback_tz_offset_hours: float = 0.0,
) -> int:
    fallback_tz = timezone(timedelta(hours=fallback_tz_offset_hours))
    by_day = _collect_track_points_by_day(gpx, fallback_tz)
    if not by_day:
        return 0

    max_gap = timedelta(minutes=max_gap_min)
    filled = 0

    for p in photos:
        if p.lat is not None or p.dt is None:
            continue

        if p.dt.tzinfo is not None:
            p_aware = p.dt
        else:
            p_aware = p.dt.replace(tzinfo=fallback_tz)

        photo_tz = p.tz or fallback_tz
        local_date = p_aware.astimezone(photo_tz).date()

        day_pts = by_day.get(local_date, [])
        if not day_pts:
            for offset in (-1, 1):
                alt = local_date + timedelta(days=offset)
                if alt in by_day:
                    day_pts = by_day[alt]
                    break
        if not day_pts:
            continue

        times = [t for t, _, _ in day_pts]
        idx = bisect.bisect_left(times, p_aware)

        if idx == 0:
            t, la, lo = day_pts[0]
            if abs(t - p_aware) <= max_gap:
                p.lat, p.lon, p.gps_source = la, lo, "gpx"
                filled += 1
        elif idx == len(day_pts):
            t, la, lo = day_pts[-1]
            if abs(t - p_aware) <= max_gap:
                p.lat, p.lon, p.gps_source = la, lo, "gpx"
                filled += 1
        else:
            t0, la0, lo0 = day_pts[idx - 1]
            t1, la1, lo1 = day_pts[idx]
            total = (t1 - t0).total_seconds()
            if total <= 0:
                p.lat, p.lon, p.gps_source = la0, lo0, "gpx"
                filled += 1
                continue
            if abs(t0 - p_aware) <= max_gap or abs(t1 - p_aware) <= max_gap:
                frac = (p_aware - t0).total_seconds() / total
                p.lat = la0 + (la1 - la0) * frac
                p.lon = lo0 + (lo1 - lo0) * frac
                p.gps_source = "gpx"
                filled += 1

    return filled


# ============================================================
#  Этап 3: кластеризация
# ============================================================

def cluster_photos(photos: list[Photo], radius_m: float) -> list[Cluster]:
    clusters: list[Cluster] = []
    no_loc: list[Photo] = []
    for p in photos:
        if p.lat is None:
            no_loc.append(p)
            continue
        placed = False
        for c in clusters:
            if c.lat is not None and haversine(c.lat, c.lon, p.lat, p.lon) < radius_m:
                c.photos.append(p)
                c.recompute()
                placed = True
                break
        if not placed:
            clusters.append(Cluster(photos=[p], lat=p.lat, lon=p.lon))
    if no_loc:
        clusters.append(Cluster(photos=no_loc))

    def _cluster_key(c: Cluster):
        dts = [p.dt for p in c.photos if p.dt]
        if not dts:
            return datetime.min.replace(tzinfo=timezone.utc)
        aware = [d if d.tzinfo else d.replace(tzinfo=timezone.utc) for d in dts]
        return min(aware)

    clusters.sort(key=_cluster_key)
    return clusters


# ============================================================
#  Этап 4: геокодинг
# ============================================================

_nominatim_cache: dict = {}


def reverse_geocode(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None:
        return None
    key = (round(lat, 2), round(lon, 2))
    if key in _nominatim_cache:
        return _nominatim_cache[key]
    try:
        r = httpx.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json",
                    "zoom": 12, "accept-language": "ru"},
            headers={"User-Agent": "travel-diary/1.0 (personal use)"},
            timeout=10,
        )
        addr = r.json().get("address", {})
        name = (addr.get("city") or addr.get("town") or addr.get("village")
                or addr.get("municipality") or addr.get("county"))
        country = addr.get("country")
        result = ", ".join(x for x in (name, country) if x) or None
    except Exception:
        result = None
    _nominatim_cache[key] = result
    time.sleep(1.1)
    return result


# ============================================================
#  Изображения
# ============================================================

try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS


def image_bytes_for_vision(path: Path, max_side: int) -> bytes:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def export_photo(src: Path, dst: Path, max_side: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    img.save(dst, format="JPEG", quality=85, optimize=True)


def _phash_value(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size, size), _LANCZOS)
    # Pillow 12+ ругается на getdata(); get_flattened_data — его замена.
    if hasattr(g, "get_flattened_data"):
        pixels = list(g.get_flattened_data())
    else:
        pixels = list(g.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for p in pixels:
        bits = (bits << 1) | (1 if p > avg else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedupe_and_diversify(
    photos: list[Photo],
    max_n: int,
    time_window_s: int = 5,
    hash_threshold: int = 10,
    max_per_cluster: int = 2,
) -> list[Photo]:
    """Возвращает до max_n разных фото по времени/хешу/кластеру."""
    if not photos:
        return []

    def _key(p: Photo):
        dt = p.dt
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    photos = sorted(photos, key=_key)

    by_time: list[Photo] = []
    last_dt: datetime | None = None
    for p in photos:
        if last_dt is not None:
            p_dt = p.dt if p.dt.tzinfo else p.dt.replace(tzinfo=timezone.utc)
            l_dt = last_dt if last_dt.tzinfo else last_dt.replace(tzinfo=timezone.utc)
            gap = (p_dt - l_dt).total_seconds()
            if 0 <= gap < time_window_s:
                continue
        by_time.append(p)
        last_dt = p.dt

    after_hash: list[Photo] = []
    hashes: list[int] = []
    for p in by_time:
        h = p.phash()
        if h is None:
            after_hash.append(p)
            continue
        if any(_hamming(h, old) < hash_threshold for old in hashes):
            continue
        hashes.append(h)
        after_hash.append(p)

    if len(after_hash) <= max_n:
        return after_hash

    by_cluster: dict = defaultdict(list)
    for p in after_hash:
        if p.lat is not None and p.lon is not None:
            key = (round(p.lat, 3), round(p.lon, 3))
        else:
            key = ("no_loc",)
        by_cluster[key].append(p)

    for k in by_cluster:
        by_cluster[k] = by_cluster[k][:max_per_cluster]

    iters = {k: iter(v) for k, v in by_cluster.items()}
    picked: list[Photo] = []
    while len(picked) < max_n and iters:
        exhausted = []
        for k, it in list(iters.items()):
            if len(picked) >= max_n:
                break
            try:
                picked.append(next(it))
            except StopIteration:
                exhausted.append(k)
        for k in exhausted:
            del iters[k]

    picked.sort(key=_key)
    return picked[:max_n]


# ============================================================
#  Промпты
# ============================================================

VISION_PROMPT = (
    "Ты рассматриваешь фотографии из личного путешествия. "
    "Опиши в 2–4 предложениях, что на них видно: обстановка, свет, люди, детали. "
    "Пиши живым языком, без списков, без вступлений вроде «на фото изображено». "
    "Если несколько кадров про одно место — расскажи про место в целом.\n"
    "Важно: если кадр снят через окно транспортного средства (самолёта, "
    "поезда, автобуса, машины), так и напиши: «вид из окна самолёта», "
    "«вид из окна поезда». Не интерпретируй облака, туман или засветку как "
    "снег или зимний пейзаж без других явных признаков зимы."
)

TITLE_SYSTEM = (
    "Ты — редактор тревел-журнала. Отвечай ровно тем, о чём просят, "
    "без пояснений и кавычек."
)

DAY_INTRO_SYSTEM = (
    "Ты — писатель-путешественник. Напиши короткое (2–4 предложения) вступление "
    "к дню поездки: где ты был, что делал, общее настроение. От первого лица, "
    "без заголовков и списков. Пиши ТОЛЬКО на русском языке, без иноязычных вставок."
)

LOCATION_NARRATIVE_SYSTEM = (
    "Ты — писатель-путешественник. Пишешь небольшой очерк (120–220 слов) об одной "
    "точке маршрута — от первого лица, с деталями, атмосферой. Опирайся на "
    "визуальные заметки и справку. Не перечисляй факты списком, вплетай их. "
    "Без заголовков. Пиши ТОЛЬКО на русском языке, без иноязычных вставок."
)

ROUTE_OVERVIEW_SYSTEM = (
    "Ты — писатель-путешественник. Составь краткий (3–5 предложений) обзор всего "
    "маршрута: откуда куда, через какие места, что было главным. От первого лица, "
    "без заголовков и списков. Пиши ТОЛЬКО на русском языке."
)

INTRO_SYSTEM = (
    "Ты — писатель-путешественник. Напиши вступление к рассказу о поездке: "
    "3–5 предложений, задающих тон. Без заголовков и списков. "
    "Пиши ТОЛЬКО на русском языке, без иноязычных вставок."
)

FINALE_SYSTEM = (
    "Ты — писатель-путешественник. Напиши лирический финал рассказа о поездке: "
    "3–5 предложений с послевкусием. Без заголовков. "
    "Пиши ТОЛЬКО на русском языке, без иноязычных вставок."
)

PROOFREAD_SYSTEM = (
    "Ты — литературный редактор. Тебе дают фрагмент русскоязычного текста, "
    "в котором могут быть случайные вставки на других языках и опечатки.\n"
    "Правила:\n"
    "1. Любое слово латиницей (t-shirt, balustrade, masts, cafe, showroom, "
    "Tissot) замени русским эквивалентом (футболка, балюстрада, мачты, кафе, "
    "шоу-рум, Тиссо). Если это имя бренда или название — оставь, но проверь, "
    "что оно написано корректно и не искажено.\n"
    "2. Иероглифы (китайские, японские и любые другие) замени осмысленными "
    "русскими словами по контексту.\n"
    "3. Транслит (например «paлto») исправь на нормальное написание («пальто»).\n"
    "4. Исправь опечатки и согласование.\n"
    "5. Сохрани авторский стиль, тон, структуру абзацев, имена собственные и числа.\n"
    "Верни только исправленный текст, без комментариев, без кавычек, "
    "без вводных фраз вроде «Вот исправленный текст»."
)


# ============================================================
#  Обёртки над LLM
# ============================================================

def describe_cluster(vision, cluster: Cluster, max_images: int, max_side: int) -> str:
    imgs: list[bytes] = []
    for p in cluster.photos[:max_images]:
        try:
            imgs.append(image_bytes_for_vision(p.path, max_side))
        except Exception as e:
            print(f"      ! {p.path.name}: {e}", file=sys.stderr)
    if not imgs:
        return ""
    prompt = VISION_PROMPT
    if cluster.place:
        prompt += f"\nКонтекст: предположительно это {cluster.place}."
    return vision.vision(imgs, prompt)


def make_cluster_title(text, cluster: Cluster) -> str:
    user = (
        f"Место: {cluster.place or 'без названия'}\n"
        f"Описание: {cluster.description or 'нет'}\n\n"
        "Придумай короткий (2–4 слова) цепляющий заголовок для этой точки "
        "маршрута на русском языке. Без кавычек и точки в конце."
    )
    fallback = cluster.place or "Место"
    t = text.text(TITLE_SYSTEM, user, reduce_source=fallback).strip()
    return t.strip('"').strip("«»").strip(".") or fallback


def write_day_intro(text, day: Day, total_km_in_day: float) -> str:
    parts = [f"Дата: {day.date.isoformat()}"]
    if total_km_in_day:
        parts.append(f"Пройдено за день: {total_km_in_day:.1f} км")
    stops = [c.place for c in day.major_clusters if c.place]
    if stops:
        parts.append("Остановки: " + " → ".join(stops))
    parts.append("Заметки о локациях:")
    for c in day.major_clusters:
        bits = [b for b in (c.place, (c.description or "")[:200]) if b]
        parts.append("- " + " — ".join(bits))
    fallback = (day.places_summary
                or f"День {day.date.strftime('%d.%m')}.")
    return text.text(DAY_INTRO_SYSTEM, "\n".join(parts), reduce_source=fallback)


def write_location_narrative(text, cluster: Cluster, day: Day) -> str:
    parts = [f"Дата: {day.date.isoformat()}"]
    if cluster.place:
        parts.append(f"Место: {cluster.place}")
    t0, t1 = cluster.time_start, cluster.time_end
    if t0 and t1:
        parts.append(f"Время съёмки: {t0.strftime('%H:%M')}–{t1.strftime('%H:%M')}")
    parts.append(f"Кадров: {len(cluster.photos)}")
    if cluster.wiki:
        parts.append(f"Справка: {cluster.wiki[:400]}")
    if cluster.description:
        parts.append(f"Что видно: {cluster.description}")
    fallback = f"{cluster.place or 'Место'}, {day.date.strftime('%d.%m.%Y')}."
    return text.text(LOCATION_NARRATIVE_SYSTEM, "\n".join(parts),
                     reduce_source=fallback)


def write_route_overview(text, days: list[Day], total_km: float) -> str:
    lines = [f"Длительность: {len(days)} дней, всего {total_km:.0f} км"]
    for i, d in enumerate(days, 1):
        stops = [c.place for c in d.major_clusters if c.place]
        km = f"{d.distance_km:.0f} км" if d.distance_km else "—"
        stops_str = ", ".join(stops) if stops else "без локаций"
        lines.append(f"День {i} ({d.date.strftime('%d.%m')}, {km}): {stops_str}")
    fallback = f"Маршрут длиной {len(days)} дней и {total_km:.0f} км."
    return text.text(ROUTE_OVERVIEW_SYSTEM, "\n".join(lines),
                     reduce_source=fallback)


def write_intro(text, days: list[Day], total_km: float) -> str:
    geo = [f"{d.date.strftime('%d.%m')}: {d.places_summary}"
           for d in days if d.places_summary]
    user = (
        f"Длительность: {len(days)} дн.\n"
        f"Всего пройдено: {total_km:.0f} км\n"
        f"География:\n" + "\n".join(geo)
    )
    fallback = f"{len(days)} дней пути, {total_km:.0f} километров."
    return text.text(INTRO_SYSTEM, user, reduce_source=fallback)


def write_finale(text, days: list[Day], total_km: float) -> str:
    user = (
        f"Поездка длилась {len(days)} дней, всего {total_km:.0f} км. "
        f"Последний день — {days[-1].date.strftime('%d.%m')} "
        f"({days[-1].places_summary or 'без локаций'})."
    )
    fallback = f"Путешествие закончилось. {len(days)} дней, {total_km:.0f} км."
    return text.text(FINALE_SYSTEM, user, reduce_source=fallback)


def proofread_text(proofread_runner, original: str) -> str:
    if not original or not original.strip():
        return original
    result = proofread_runner.text(PROOFREAD_SYSTEM, original, reduce_source="")
    return result if result and result.strip() else original


# ============================================================
#  Постобработка
# ============================================================

def dedup_wiki_texts(days: list[Day]) -> None:
    """Одинаковый текст Википедии оставляем только на первом слайде."""
    seen: set[int] = set()
    for day in days:
        for c in day.clusters:
            if not c.wiki:
                continue
            h = hash(c.wiki[:200])
            if h in seen:
                c.wiki = None
            else:
                seen.add(h)


# ============================================================
#  Классификация
# ============================================================

def classify_day_clusters(
    day: Day,
    significant_min_photos: int,
    require_wiki: bool,
) -> None:
    majors: list[Cluster] = []
    minors: list[Cluster] = []

    for c in day.clusters:
        if c.place is None:
            c.significant = False
            minors.append(c)
            continue
        if require_wiki:
            c.significant = c.wiki is not None
        else:
            c.significant = (len(c.photos) >= significant_min_photos) \
                            or (c.wiki is not None)
        (majors if c.significant else minors).append(c)

    if not majors and day.clusters:
        largest = max(day.clusters, key=lambda c: len(c.photos))
        largest.significant = True
        majors = [largest]
        minors = [c for c in day.clusters if c is not largest]

    day.major_clusters = majors
    day.minor_clusters = minors


# ============================================================
#  Выбор обложки
# ============================================================

def pick_cover(days: list[Day], dedup_kwargs: dict) -> Photo | None:
    """Обложка из середины поездки, предпочтительно дневные кадры."""
    if not days:
        return None
    mid = len(days) // 2
    order = sorted(range(len(days)), key=lambda i: abs(i - mid))
    for idx in order:
        day = days[idx]
        day_photos = [p for p in day.photos
                      if p.dt and 10 <= p.dt.hour <= 18]
        pool = day_photos or day.photos
        if not pool:
            continue
        picked = dedupe_and_diversify(pool, max_n=1, **dedup_kwargs)
        if picked:
            return picked[0]
    return None


# ============================================================
#  CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="HTML-дневник путешествия")
    ap.add_argument("photos", type=Path, help="папка с фотографиями")
    ap.add_argument("-o", "--out", type=Path, default=Path("diary"),
                    help="папка результата (по умолчанию ./diary)")
    ap.add_argument("-c", "--config", type=Path, default=Path("config.yaml"),
                    help="путь к config.yaml")
    ap.add_argument("-g", "--gpx", type=Path, default=None,
                    help="файл GPX с треком")
    ap.add_argument("--skip-vision", action="store_true",
                    help="не описывать кадры (быстрее, дешевле)")
    ap.add_argument("--skip-proofread", action="store_true",
                    help="не прогонять тексты через корректора")
    ap.add_argument("--single-slide-days", action="store_true",
                    help="старый режим: один слайд на день, без локаций")
    ap.add_argument("--max-days", type=int, default=None,
                    help="обработать только первые N дней (отладка)")
    ap.add_argument("--no-phase", action="store_true",
                    help="отключить фазовый режим Ollama")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


# ============================================================
#  main
# ============================================================

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "ollama",
                  "charset_normalizer", "filelock", "asyncio", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not args.config.exists():
        sys.exit(f"Нет файла конфигурации: {args.config}")
    cfg = yaml.safe_load(args.config.read_text())

    if args.no_phase:
        cfg.setdefault("runtime", {})["phase_mode"] = False
        cfg["runtime"]["preload"] = False
        cfg["runtime"]["unload_between_phases"] = False

    registry = Registry(cfg)
    vision = registry.role("vision")
    text = registry.role("text")
    proofread = registry.role("proofread") if registry.has_role("proofread") else None

    proc = cfg.get("processing", {})
    vision_max_side = proc.get("vision_max_side", 768)
    export_max_side = proc.get("export_max_side", 1600)
    cluster_radius = proc.get("cluster_radius_m", 400)
    max_img = proc.get("max_images_per_cluster", 4)
    do_geocode = proc.get("use_reverse_geocode", True)
    search_cfg = cfg.get("search", {})

    use_mtime_fallback = proc.get("use_file_mtime_fallback", False)
    photo_tz_offset = proc.get("photo_tz_offset_hours", 0.0)

    max_photos_per_day = proc.get("max_photos_per_day", 6)
    max_photos_per_location = proc.get("max_photos_per_location", 4)
    dedup_time_window_s = proc.get("dedup_time_window_s", 5)
    dedup_hash_threshold = proc.get("dedup_hash_threshold", 10)
    max_photos_per_cluster = proc.get("max_photos_per_cluster", 2)

    significant_min_photos = proc.get("significant_min_photos", 3)
    significant_require_wiki = proc.get("significant_require_wiki", False)
    day_intro_min_locations = proc.get("day_intro_min_locations", 2)
    generate_route_overview = proc.get("generate_route_overview", True)

    transit_min_clusters = proc.get("transit_min_clusters", 2)
    transit_min_photos = proc.get("transit_min_photos", 3)

    gps_interp = proc.get("gps_interpolation_from_gpx", True)
    gps_max_gap = proc.get("gps_interpolation_max_gap_min", 30)

    single_slide_days = args.single_slide_days

    # ---------- 1. Метаданные ----------
    print(f"[1/7] Собираю метаданные из {args.photos}")
    if not args.photos.exists():
        sys.exit(f"Папка не найдена: {args.photos}")

    fallback_tz = timezone(timedelta(hours=photo_tz_offset))
    photos, stats = collect_metadata(
        args.photos, use_mtime_fallback, fallback_tz)
    print(f"      Файлов найдено: {stats['files']}")
    print(f"      С датой:        {len(photos)}")
    if stats["no_dt"]:
        print(f"      Без даты:       {stats['no_dt']} (пропущено)")
    if stats["from_mtime"]:
        print(f"      Из mtime:       {stats['from_mtime']}")
    if stats["no_tz"]:
        print(f"      Без tz в EXIF:  {stats['no_tz']} (использую fallback "
              f"{photo_tz_offset:+.0f}:00)")
    if stats["no_gps"]:
        print(f"      Без GPS:        {stats['no_gps']}")
    if not photos:
        sys.exit("Нет фотографий с датой съёмки.")

    gpx = load_gpx(args.gpx) if args.gpx and args.gpx.exists() else None

    # ---------- 2. Интерполяция GPS ----------
    if gpx and gps_interp:
        n_gps = interpolate_gps_from_gpx(photos, gpx, gps_max_gap, photo_tz_offset)
        if n_gps:
            print(f"      Интерполировано координат из GPX: {n_gps}")

    # ---------- 3. Группировка по дням и кластеризация ----------
    print("[2/7] Группирую по дням и локациям")
    by_day = defaultdict(list)
    for p in photos:
        by_day[p.day].append(p)

    days: list[Day] = []
    total_km = 0.0
    for d in sorted(by_day):
        day = Day(date=d, photos=by_day[d])
        if gpx:
            day.distance_km = track_distance_for_day(gpx, d, fallback_tz)
            total_km += day.distance_km
        day.clusters = cluster_photos(day.photos, cluster_radius)
        days.append(day)

    if args.max_days:
        days = days[: args.max_days]
        total_km = sum(d.distance_km for d in days)

    n_clusters = sum(len(d.clusters) for d in days)
    print(f"      Дней: {len(days)}, кластеров: {n_clusters}, "
          f"суммарно {total_km:.0f} км")

    # ---------- 4. Геокодинг + Википедия ----------
    print("[3/7] Геокодинг и Википедия")
    for day in days:
        for c in day.clusters:
            if do_geocode and c.lat is not None:
                c.place = reverse_geocode(c.lat, c.lon)
            if search_cfg.get("enabled") and (c.place or c.lat is not None):
                c.wiki = wiki_summary(
                    c.place,
                    lat=c.lat,
                    lon=c.lon,
                    lang=search_cfg.get("language", "ru"),
                    max_chars=search_cfg.get("max_chars", 600),
                    timeout=search_cfg.get("timeout", 8),
                )
                if c.wiki:
                    tag = c.place or f"{c.lat:.3f},{c.lon:.3f}"
                    print(f"      · {tag}: {c.wiki[:60]}…")

    dedup_wiki_texts(days)

    # ---------- 5. ФАЗА A: VISION ----------
    if not args.skip_vision:
        print("[4/7] Фаза A: описания фотографий")
        phase_a = registry.begin_phase("vision")
        try:
            total = sum(len(d.clusters) for d in days)
            n = 0
            for day in days:
                for c in day.clusters:
                    n += 1
                    print(f"      [{n}/{total}] {day.date} · "
                          f"{len(c.photos)} фото · {c.place or '—'}")
                    c.description = describe_cluster(
                        vision, c, max_img, vision_max_side)
        finally:
            registry.end_phase(phase_a)
    else:
        print("[4/7] Пропускаю vision-фазу (--skip-vision)")

    # ---------- 6. ФАЗА B: TEXT + PROOFREAD ----------
    print("[5/7] Фаза B: тексты и вычитка")
    route_overview = intro = finale = ""
    do_proof = proofread is not None and not args.skip_proofread
    phase_roles = ["text"] + (["proofread"] if do_proof else [])
    phase_b = registry.begin_phase(*phase_roles)
    try:
        print("      заголовки локаций…")
        for day in days:
            for c in day.clusters:
                c.title = (make_cluster_title(text, c) if c.description
                           else (c.place or "Кадры"))

        if single_slide_days:
            for day in days:
                day.major_clusters = day.clusters
                day.minor_clusters = []
                for c in day.clusters:
                    c.significant = True
        else:
            for day in days:
                classify_day_clusters(
                    day, significant_min_photos, significant_require_wiki)

        for day in days:
            places = [c.place for c in day.major_clusters if c.place]
            places = dedup_consecutive(places)
            day.places_summary = " → ".join(places)
            day.title = day.places_summary or day.date.strftime("%d.%m.%Y")

        if not single_slide_days:
            print("      рассказы по локациям…")
            for day in days:
                for c in day.major_clusters:
                    if c.description or c.wiki:
                        print(f"        · {day.date} · {c.place or '—'}")
                        c.narrative = write_location_narrative(text, c, day)

        if not single_slide_days:
            print("      вступления дней…")
            for day in days:
                if len(day.major_clusters) >= day_intro_min_locations:
                    day.day_intro = write_day_intro(text, day, day.distance_km)
                elif day.major_clusters:
                    c = day.major_clusters[0]
                    if not c.narrative:
                        c.narrative = write_location_narrative(text, c, day)

        if generate_route_overview and len(days) > 1:
            print("      обзор маршрута…")
            route_overview = write_route_overview(text, days, total_km)

        print("      пролог и финал…")
        intro = write_intro(text, days, total_km)
        finale = write_finale(text, days, total_km)

        if do_proof:
            print("      вычитка текстов…")
            for day in days:
                if day.day_intro:
                    day.day_intro = proofread_text(proofread, day.day_intro)
                for c in day.major_clusters:
                    if c.narrative:
                        c.narrative = proofread_text(proofread, c.narrative)
                    if c.description:
                        c.description = proofread_text(proofread, c.description)
                for c in day.minor_clusters:
                    if c.description:
                        c.description = proofread_text(proofread, c.description)
            intro = proofread_text(proofread, intro)
            finale = proofread_text(proofread, finale)
            route_overview = proofread_text(proofread, route_overview)
    finally:
        registry.end_phase(phase_b)

    # ---------- 7. Экспорт фото и сборка HTML ----------
    print("[6/7] Экспорт фотографий")
    out = args.out
    photos_dir = out / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    def export(p: Photo) -> str:
        dst = photos_dir / p.path.name
        if not dst.exists():
            try:
                export_photo(p.path, dst, export_max_side)
            except Exception as e:
                print(f"      ! {p.path.name}: {e}", file=sys.stderr)
                return ""
        return f"photos/{dst.name}"

    slides: list[dict] = []

    dedup_kwargs = {
        "time_window_s": dedup_time_window_s,
        "hash_threshold": dedup_hash_threshold,
        "max_per_cluster": 1,
    }
    cover_photo_obj = pick_cover(days, dedup_kwargs)
    cover_photo = export(cover_photo_obj) if cover_photo_obj else ""

    slides.append({
        "kind": "cover",
        "title": cfg.get("output", {}).get("title", "Моё путешествие"),
        "subtitle": (f"{days[0].date.strftime('%d.%m.%Y')} — "
                     f"{days[-1].date.strftime('%d.%m.%Y')}"),
        "photo": cover_photo,
        "meta": f"{len(days)} дней · {total_km:.0f} км · {len(photos)} кадров",
    })

    slides.append({"kind": "intro", "title": "Пролог", "text": intro})

    if route_overview:
        stops: list[dict] = []
        for i, d in enumerate(days, 1):
            places = [c.place for c in d.major_clusters if c.place]
            places = dedup_consecutive(places)
            if not places:
                continue
            stops.append({
                "day": i,
                "date": d.date.strftime("%d.%m"),
                "place": " → ".join(places),
                "km": f"{d.distance_km:.0f} км" if d.distance_km else "",
            })
        slides.append({
            "kind": "overview",
            "title": "Маршрут",
            "narrative": route_overview,
            "stops": stops,
        })

    for i, day in enumerate(days, 1):
        if single_slide_days:
            day_photos = dedupe_and_diversify(
                day.photos, max_n=max_photos_per_day,
                time_window_s=dedup_time_window_s,
                hash_threshold=dedup_hash_threshold,
                max_per_cluster=max_photos_per_cluster,
            )
            photos_html = [x for x in (export(p) for p in day_photos) if x]
            slides.append({
                "kind": "day",
                "index": i,
                "date": day.date.strftime("%d.%m.%Y"),
                "title": day.title,
                "meta": f"{day.distance_km:.1f} км" if day.distance_km else "",
                "narrative": day.day_intro or "",
                "clusters": [
                    {"title": c.title, "place": c.place,
                     "description": c.description, "wiki": c.wiki,
                     "narrative": c.narrative}
                    for c in day.clusters
                ],
                "photos": photos_html,
            })
            continue

        if day.day_intro and len(day.major_clusters) >= day_intro_min_locations:
            intro_photos = dedupe_and_diversify(
                day.photos, max_n=max_photos_per_day,
                time_window_s=dedup_time_window_s,
                hash_threshold=dedup_hash_threshold,
                max_per_cluster=1,
            )
            intro_photos_html = [x for x in (export(p) for p in intro_photos) if x]
            slides.append({
                "kind": "day-intro",
                "index": i,
                "date": day.date.strftime("%d.%m.%Y"),
                "title": day.title,
                "meta": f"{day.distance_km:.1f} км" if day.distance_km else "",
                "narrative": day.day_intro,
                "photos": intro_photos_html,
            })

        for c in day.major_clusters:
            picked_photos = dedupe_and_diversify(
                c.photos, max_n=max_photos_per_location,
                time_window_s=dedup_time_window_s,
                hash_threshold=dedup_hash_threshold,
                max_per_cluster=1,
            )
            photos_html = [x for x in (export(p) for p in picked_photos) if x]
            slides.append({
                "kind": "location",
                "index": i,
                "date": day.date.strftime("%d.%m.%Y"),
                "day_title": day.title,
                "title": c.title,
                "place": c.place,
                "meta": f"{day.distance_km:.1f} км" if day.distance_km else "",
                "narrative": c.narrative,
                "description": c.description,
                "wiki": c.wiki,
                "photos": photos_html,
            })

        minor_with_photos = [c for c in day.minor_clusters if c.photos]
        total_minor_photos = sum(len(c.photos) for c in minor_with_photos)
        if (len(minor_with_photos) >= transit_min_clusters
                and total_minor_photos >= transit_min_photos):
            transit_pool: list[Photo] = []
            for c in minor_with_photos:
                transit_pool.extend(c.photos)
            picked = dedupe_and_diversify(
                transit_pool, max_n=max_photos_per_day,
                time_window_s=dedup_time_window_s,
                hash_threshold=dedup_hash_threshold,
                max_per_cluster=1,
            )
            photos_html = [x for x in (export(p) for p in picked) if x]
            slides.append({
                "kind": "transit",
                "index": i,
                "date": day.date.strftime("%d.%m.%Y"),
                "clusters": [
                    {"place": c.place or "Без локации",
                     "description": c.description}
                    for c in minor_with_photos
                ],
                "photos": photos_html,
            })

    slides.append({"kind": "finale", "title": "Послесловие", "text": finale})

    print("[7/7] Сборка HTML")
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=True)
    tpl = env.get_template("slides.html.j2")
    html = tpl.render(
        title=cfg.get("output", {}).get("title", "Моё путешествие"),
        theme=cfg.get("output", {}).get("theme", "dark"),
        slides=slides,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")

    print("\n" + registry.report())
    print(f"\nГотово: {(out / 'index.html').resolve()}")
    print(f"Слайдов: {len(slides)}")


if __name__ == "__main__":
    main()