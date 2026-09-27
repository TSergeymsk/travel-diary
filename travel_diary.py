#!/usr/bin/env python3
"""
travel_diary.py — генерирует HTML-дневник путешествия.

Пайплайн:
  1. EXIF + GPX           → дни, кластеры, расстояния
                            (+ интерполяция GPS из трека для фото без геометок)
  2. Nominatim + Википедия → названия мест, краткие справки
  3. Фаза VISION          → описания всех кластеров
  4. Фаза TEXT+PROOFREAD  → заголовки, рассказы, вступление, финал + вычитка
  5. Экспорт фото + HTML  (с дедупликацией и диверсификацией кадров)
"""
from __future__ import annotations

import argparse
import bisect
import io
import logging
import math
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
    gps_source: str | None = None   # 'exif' | 'gpx' | None — откуда координаты

    @property
    def day(self):
        return self.dt.date() if self.dt else None


@dataclass
class Cluster:
    photos: list[Photo] = field(default_factory=list)
    lat: float | None = None
    lon: float | None = None
    place: str | None = None
    wiki: str | None = None
    title: str = ""
    description: str = ""

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
    narrative: str = ""
    places_summary: str = ""


# ============================================================
#  Чтение фотографий
# ============================================================

EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff"}


def read_photo(path: Path) -> Photo | None:
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
                break
            except ValueError:
                pass

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


def load_photos(root: Path) -> list[Photo]:
    out: list[Photo] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in EXTS:
            ph = read_photo(path)
            if ph and ph.dt:
                out.append(ph)
    out.sort(key=lambda x: x.dt)
    return out


# ============================================================
#  GPX
# ============================================================

def load_gpx(path: Path):
    with path.open() as f:
        return gpxpy.parse(f)


def _collect_track_points_by_day(gpx, photo_tz: timezone) -> dict:
    """Индексирует точки трека по локальной дате в часовом поясе фотографий.
       Все времена приводятся к aware-виду (naive → UTC)."""
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
    photo_tz_offset_hours: float = 0.0,
) -> int:
    """
    Заполняет координаты фото без GPS из ближайших точек GPX по времени.

    photo_tz_offset_hours — смещение часового пояса, в котором сняты фото
    (EXIF DateTimeOriginal всегда naive local). GPX-времена обычно tz-aware UTC.
    Чтобы их корректно сравнить, наивное локальное время фото переводится
    в aware через это смещение.
    """
    photo_tz = timezone(timedelta(hours=photo_tz_offset_hours))
    by_day = _collect_track_points_by_day(gpx, photo_tz)
    if not by_day:
        return 0

    max_gap = timedelta(minutes=max_gap_min)
    filled = 0

    for p in photos:
        if p.lat is not None or p.dt is None:
            continue

        # делаем dt фото aware в его локальной зоне
        p_aware = (p.dt if p.dt.tzinfo
                   else p.dt.replace(tzinfo=photo_tz))

        day_pts = by_day.get(p.dt.date(), [])
        if not day_pts:
            for offset in (-1, 1):
                alt = p.dt.date() + timedelta(days=offset)
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
#  Кластеризация и геокодирование
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
    clusters.sort(key=lambda c: min(p.dt for p in c.photos))
    return clusters


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
    time.sleep(1.1)  # вежливость к OSM
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


# ---------- Перцептивный хеш ----------

def _phash(img: Image.Image, size: int = 8) -> int:
    """Average-hash: 64-битное число, устойчивое к мелким изменениям яркости/масштаба."""
    g = img.convert("L").resize((size, size), _LANCZOS)
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
    """
    Возвращает до max_n разных фото:
      1) убирает снимки, сделанные подряд в пределах time_window_s секунд;
      2) убирает визуально похожие (перцептивный хеш, хемминг < hash_threshold);
      3) ограничивает вклад одного географического кластера (max_per_cluster);
      4) распределяет выборку round-robin по кластерам.
    """
    if not photos:
        return []

    photos = sorted(photos, key=lambda p: p.dt)

    # 1) по времени
    by_time: list[Photo] = []
    last_dt: datetime | None = None
    for p in photos:
        if last_dt is not None:
            gap = (p.dt - last_dt).total_seconds()
            if 0 <= gap < time_window_s:
                continue
        by_time.append(p)
        last_dt = p.dt

    # 2) по перцептивному хешу
    after_hash: list[Photo] = []
    hashes: list[int] = []
    for p in by_time:
        try:
            img = Image.open(p.path)
            img = ImageOps.exif_transpose(img)
            h = _phash(img)
            if any(_hamming(h, old) < hash_threshold for old in hashes):
                continue
            hashes.append(h)
            after_hash.append(p)
        except Exception:
            after_hash.append(p)

    if len(after_hash) <= max_n:
        return after_hash

    # 3) группируем по кластерам
    by_cluster: dict = defaultdict(list)
    for p in after_hash:
        if p.lat is not None and p.lon is not None:
            key = (round(p.lat, 3), round(p.lon, 3))
        else:
            key = ("no_loc",)
        by_cluster[key].append(p)

    for k in by_cluster:
        by_cluster[k] = by_cluster[k][:max_per_cluster]

    # 4) round-robin по кластерам
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

    picked.sort(key=lambda p: p.dt)
    return picked[:max_n]


# ============================================================
#  Промпты
# ============================================================

VISION_PROMPT = (
    "Ты рассматриваешь фотографии из личного путешествия. "
    "Опиши в 2–4 предложениях, что на них видно: обстановка, свет, люди, детали. "
    "Пиши живым языком, без списков, без вступлений вроде «на фото изображено». "
    "Если несколько кадров про одно место — расскажи про место в целом."
)

TITLE_SYSTEM = (
    "Ты — редактор тревел-журнала. Отвечай ровно тем, о чём просят, "
    "без пояснений и кавычек."
)

DAY_SYSTEM = (
    "Ты — писатель-путешественник. Пишешь тёплый, образный, слегка ироничный "
    "дневник поездки от первого лица. Вплетай детали из заметок в повествование, "
    "не перечисляй их списком. 2–3 абзаца, 250–350 слов, без заголовков. "
    "Пиши ТОЛЬКО на русском языке, без иноязычных вставок."
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
    "в котором могут быть случайные вставки на других языках (китайские "
    "иероглифы, английские слова, латиница, транслит) и опечатки. "
    "Замени иноязычные вставки осмысленными русскими словами по контексту. "
    "Исправь явные опечатки и согласование. Сохрани авторский стиль, тон, "
    "структуру абзацев, имена собственные и числа. "
    "Верни только исправленный текст, без комментариев, без кавычек, "
    "без вводных фраз вроде «Вот исправленный текст»."
)


# ============================================================
#  Обёртки над RoleRunner
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


def write_day_narrative(text, day: Day) -> str:
    parts = [f"Дата: {day.date.isoformat()}"]
    if day.distance_km:
        parts.append(f"Пройдено за день: {day.distance_km:.1f} км")
    if day.places_summary:
        parts.append(f"Места: {day.places_summary}")
    parts.append("Заметки о кадрах:")
    for c in day.clusters:
        bits = [b for b in (c.place, c.wiki, c.description) if b]
        parts.append("- " + " — ".join(bits))
    fallback = f"День {day.date.isoformat()}: {day.places_summary or 'без локаций'}."
    return text.text(DAY_SYSTEM, "\n".join(parts), reduce_source=fallback)


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
    """
    Прогон текста через корректора. При любой ошибке возвращает оригинал —
    вычитка не должна ломать финальный результат.
    """
    if not original or not original.strip():
        return original
    result = proofread_runner.text(PROOFREAD_SYSTEM, original, reduce_source="")
    return result if result and result.strip() else original


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
    # Гасим болтливые библиотеки
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

    max_photos_per_day = proc.get("max_photos_per_day", 6)
    dedup_time_window_s = proc.get("dedup_time_window_s", 5)
    dedup_hash_threshold = proc.get("dedup_hash_threshold", 10)
    max_photos_per_cluster = proc.get("max_photos_per_cluster", 2)
    gps_interp = proc.get("gps_interpolation_from_gpx", True)
    gps_max_gap = proc.get("gps_interpolation_max_gap_min", 30)
    gps_photo_tz = proc.get("photo_tz_offset_hours", 0.0)

    # ---------- 1. Фото ----------
    print(f"[1/6] Читаю фотографии из {args.photos}")
    if not args.photos.exists():
        sys.exit(f"Папка не найдена: {args.photos}")
    photos = load_photos(args.photos)
    if not photos:
        sys.exit("Нет фотографий с датой съёмки.")
    print(f"      Найдено: {len(photos)}")

    gpx = load_gpx(args.gpx) if args.gpx and args.gpx.exists() else None

    # ---------- Интерполяция GPS из GPX ----------
    if gpx and gps_interp:
        n_gps = interpolate_gps_from_gpx(photos, gpx, gps_max_gap, gps_photo_tz)
        if n_gps:
            print(f"      Интерполировано координат из GPX: {n_gps}")

    # ---------- 2. Группировка по дням и кластеризация ----------
    by_day = defaultdict(list)
    for p in photos:
        by_day[p.day].append(p)

    photo_tz = timezone(timedelta(hours=gps_photo_tz))

    days: list[Day] = []
    total_km = 0.0
    for d in sorted(by_day):
        day = Day(date=d, photos=by_day[d])
        if gpx:
            day.distance_km = track_distance_for_day(gpx, d, photo_tz)
            total_km += day.distance_km
        day.clusters = cluster_photos(day.photos, cluster_radius)
        days.append(day)

    if args.max_days:
        days = days[: args.max_days]
        total_km = sum(d.distance_km for d in days)

    # ---------- 3. Сеть: геокодинг + Википедия ----------
    print("[2/6] Геокодинг и Википедия")
    for day in days:
        for c in day.clusters:
            if do_geocode and c.lat is not None:
                c.place = reverse_geocode(c.lat, c.lon)
            if search_cfg.get("enabled") and c.place:
                toponym = c.place.split(",")[0].strip()
                c.wiki = wiki_summary(
                    toponym,
                    lang=search_cfg.get("language", "ru"),
                    max_chars=search_cfg.get("max_chars", 600),
                    timeout=search_cfg.get("timeout", 8),
                )
                if c.wiki:
                    print(f"      · {toponym}: {c.wiki[:60]}…")

    # ---------- 4. ФАЗА A: VISION ----------
    if not args.skip_vision:
        print("[3/6] Фаза A: описания фотографий")
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
        print("[3/6] Пропускаю vision-фазу (--skip-vision)")

    # ---------- 5. ФАЗА B: TEXT + PROOFREAD ----------
    print("[4/6] Фаза B: тексты и вычитка")
    intro = finale = ""
    do_proof = proofread is not None and not args.skip_proofread
    phase_roles = ["text"] + (["proofread"] if do_proof else [])
    phase_b = registry.begin_phase(*phase_roles)
    try:
        # Заголовки и сводки по дням
        for day in days:
            for c in day.clusters:
                c.title = (make_cluster_title(text, c) if c.description
                           else (c.place or "Кадры"))
            uniq = list(dict.fromkeys(c.place for c in day.clusters if c.place))
            day.places_summary = " → ".join(uniq)
            day.title = day.places_summary or day.date.strftime("%d.%m.%Y")

        # Рассказы по дням
        for day in days:
            print(f"      день {day.date}: рассказ…")
            day.narrative = write_day_narrative(text, day)

        print("      вступление…")
        intro = write_intro(text, days, total_km)
        print("      финал…")
        finale = write_finale(text, days, total_km)

        # Вычитка
        if do_proof:
            print("      вычитка текстов…")
            for day in days:
                if day.narrative:
                    day.narrative = proofread_text(proofread, day.narrative)
                for c in day.clusters:
                    if c.description:
                        c.description = proofread_text(proofread, c.description)
            intro = proofread_text(proofread, intro)
            finale = proofread_text(proofread, finale)
    finally:
        registry.end_phase(phase_b)

    # ---------- 6. Экспорт фото (с дедупликацией и диверсификацией) ----------
    print("[5/6] Экспорт фотографий")
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

    # Обложка
    cover_pool = dedupe_and_diversify(
        days[0].photos,
        max_n=1,
        time_window_s=dedup_time_window_s,
        hash_threshold=dedup_hash_threshold,
        max_per_cluster=1,
    )
    cover_photo = export(cover_pool[0]) if cover_pool else export(days[0].photos[0])
    slides.append({
        "kind": "cover",
        "title": cfg.get("output", {}).get("title", "Моё путешествие"),
        "subtitle": (f"{days[0].date.strftime('%d.%m.%Y')} — "
                     f"{days[-1].date.strftime('%d.%m.%Y')}"),
        "photo": cover_photo,
        "meta": f"{len(days)} дней · {total_km:.0f} км · {len(photos)} кадров",
    })

    slides.append({"kind": "intro", "title": "Пролог", "text": intro})

    for i, day in enumerate(days, 1):
        picked = dedupe_and_diversify(
            day.photos,
            max_n=max_photos_per_day,
            time_window_s=dedup_time_window_s,
            hash_threshold=dedup_hash_threshold,
            max_per_cluster=max_photos_per_cluster,
        )
        day_photos = [x for x in (export(p) for p in picked) if x]
        slides.append({
            "kind": "day",
            "index": i,
            "date": day.date.strftime("%d.%m.%Y"),
            "title": day.title,
            "meta": f"{day.distance_km:.1f} км" if day.distance_km else "",
            "narrative": day.narrative,
            "clusters": [
                {"title": c.title, "place": c.place,
                 "description": c.description, "wiki": c.wiki}
                for c in day.clusters
            ],
            "photos": day_photos,
        })

    slides.append({"kind": "finale", "title": "Послесловие", "text": finale})

    # ---------- 7. HTML ----------
    print("[6/6] Сборка HTML")
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


if __name__ == "__main__":
    main()