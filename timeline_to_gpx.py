#!/usr/bin/env python3
"""
Конвертер Google Timeline JSON → GPX с сохранением временных меток.

Поддерживает:
  • старый формат Takeout:      {"locations": [{"timestamp", "latitudeE7", ...}, ...]}
  • новый формат с телефона:    {"semanticSegments": [{"timelinePath": [...], ...}, ...]}

Координаты в новом формате могут быть представлены в нескольких видах:
  "geo:55.75,37.61"
  {"latLng": "geo:55.75,37.61"}
  {"latLng": {"latitudeE7": 557500000, "longitudeE7": 376100000}}
  {"latitude": 55.75, "longitude": 37.61}
Все варианты обрабатываются рекурсивно.

На выходе — GPX с одним <trkseg> на каждый календарный день.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict


# ============================================================
#  Разбор времени
# ============================================================

def parse_iso(s: str) -> datetime | None:
    """Парсит ISO 8601 в разных вариантах (Z, +00:00, доли секунды, без них)."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        s2 = s.replace("Z", "+00:00") if s.endswith("Z") else s
        try:
            dt = datetime.fromisoformat(s2)
        except ValueError:
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
                try:
                    return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ============================================================
#  Разбор координат — универсальный
# ============================================================

def geo_to_floats(loc) -> tuple[float, float] | None:
    """Принимает строку 'geo:lat,lon', dict с latLng, dict с latitudeE7/longitudeE7
       или dict с latitude/longitude. Возвращает (lat, lon) или None."""
    if not loc:
        return None

    # 1) строка "geo:lat,lon"
    if isinstance(loc, str):
        s = loc.strip()
        if s.startswith("geo:"):
            s = s[4:]
        try:
            lat_s, lon_s = s.split(",")
            return float(lat_s), float(lon_s)
        except (ValueError, IndexError):
            return None

    # 2) словарь — рекурсивно спускаемся в latLng / координаты
    if isinstance(loc, dict):
        if "latLng" in loc:
            return geo_to_floats(loc["latLng"])
        if "latitudeE7" in loc and "longitudeE7" in loc:
            try:
                return loc["latitudeE7"] / 1e7, loc["longitudeE7"] / 1e7
            except (TypeError, KeyError):
                return None
        if "latitude" in loc and "longitude" in loc:
            try:
                return float(loc["latitude"]), float(loc["longitude"])
            except (TypeError, ValueError):
                return None
        if "location" in loc:
            return geo_to_floats(loc["location"])
    return None


# ============================================================
#  Парсеры форматов
# ============================================================

def parse_old_format(data: dict) -> list[tuple[float, float, datetime]]:
    """Старый Takeout: locations[] с timestamp и latitudeE7/longitudeE7."""
    out = []
    for entry in data.get("locations", []) or []:
        if not isinstance(entry, dict):
            continue
        try:
            lat = entry["latitudeE7"] / 1e7
            lon = entry["longitudeE7"] / 1e7
        except (KeyError, TypeError):
            continue
        dt = parse_iso(entry.get("timestamp", ""))
        if dt:
            out.append((lat, lon, dt))
    return out


def parse_new_format(data: dict) -> list[tuple[float, float, datetime]]:
    """Новый формат: semanticSegments[] с timelinePath / visit / activity / rawSignals."""
    out: list[tuple[float, float, datetime]] = []

    for seg in data.get("semanticSegments", []) or []:
        if not isinstance(seg, dict):
            continue

        # 1) точки трека
        for pt in seg.get("timelinePath", []) or []:
            if not isinstance(pt, dict):
                continue
            coords = geo_to_floats(pt.get("point"))
            dt = parse_iso(pt.get("time", ""))
            if coords and dt:
                out.append((coords[0], coords[1], dt))

        # 2) посещение места
        visit = seg.get("visit")
        if isinstance(visit, dict):
            cand = visit.get("topCandidate") or {}
            coords = (
                geo_to_floats(cand.get("placeLocation"))
                or geo_to_floats(cand.get("placeId"))
                or geo_to_floats(visit.get("placeLocation"))
            )
            dt = parse_iso(seg.get("startTime", ""))
            if coords and dt:
                out.append((coords[0], coords[1], dt))

        # 3) активность (поездка)
        act = seg.get("activity")
        if isinstance(act, dict):
            for key, tkey in (("start", "startTime"), ("end", "endTime")):
                coords = geo_to_floats(act.get(key))
                dt = parse_iso(seg.get(tkey, ""))
                if coords and dt:
                    out.append((coords[0], coords[1], dt))

        # 4) сырые сигналы (встречаются в некоторых экспортах)
        for rs in seg.get("rawSignals", []) or []:
            if not isinstance(rs, dict):
                continue
            pos = rs.get("position")
            if isinstance(pos, dict):
                coords = geo_to_floats(pos.get("LatLng") or pos.get("latLng"))
                dt = parse_iso(pos.get("timestamp", ""))
                if coords and dt:
                    out.append((coords[0], coords[1], dt))

    return out


def detect_format(data: dict) -> str:
    if "semanticSegments" in data:
        return "new"
    if "locations" in data:
        return "old"
    return "unknown"


# ============================================================
#  GPX-вывод
# ============================================================

def write_gpx(
    points: list[tuple[float, float, datetime]],
    output: Path,
    tz_offset_hours: float = 0.0,
) -> int:
    """Пишет точки в GPX, разбивая на <trkseg> по локальным дням."""
    points.sort(key=lambda p: p[2])

    # дедупликация
    seen = set()
    deduped = []
    for lat, lon, dt in points:
        key = (round(lat, 6), round(lon, 6), int(dt.timestamp()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((lat, lon, dt))

    tz = timezone(timedelta(hours=tz_offset_hours))

    by_day: dict = defaultdict(list)
    for lat, lon, dt in deduped:
        by_day[dt.astimezone(tz).date()].append((lat, lon, dt))

    with output.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<gpx version="1.1" creator="timeline_to_gpx.py" '
                'xmlns="http://www.topografix.com/GPX/1/1">\n')
        f.write('  <metadata>\n')
        f.write('    <name>Google Timeline export</name>\n')
        f.write('  </metadata>\n')

        for day in sorted(by_day):
            f.write(f'  <trk>\n    <name>{day.isoformat()}</name>\n')
            f.write('    <trkseg>\n')
            for lat, lon, dt in by_day[day]:
                iso = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                f.write(f'      <trkpt lat="{lat:.7f}" lon="{lon:.7f}">'
                        f'<time>{iso}</time></trkpt>\n')
            f.write('    </trkseg>\n  </trk>\n')

        f.write('</gpx>\n')
    return len(deduped)


# ============================================================
#  CLI
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Google Timeline JSON → GPX")
    ap.add_argument("input", type=Path, help="timeline.json")
    ap.add_argument("-o", "--output", type=Path, default=Path("timeline.gpx"))
    ap.add_argument("--tz", type=float, default=0.0,
                    help="смещение локального часового пояса в часах "
                         "(Москва = 3). Влияет только на разбивку по дням.")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"Файл не найден: {args.input}")

    with args.input.open(encoding="utf-8") as f:
        data = json.load(f)

    fmt = detect_format(data)
    print(f"Формат: {fmt}")

    if fmt == "old":
        points = parse_old_format(data)
    elif fmt == "new":
        points = parse_new_format(data)
    else:
        sys.exit("Неизвестный формат JSON — нет ни 'locations', ни 'semanticSegments'.")

    if not points:
        sys.exit("Не удалось извлечь ни одной точки с координатами и временем. "
                 "Возможно, структура JSON отличается от ожидаемой — "
                 "смотрите пример сегмента в диагностике.")

    print(f"Извлечено точек: {len(points)}")
    written = write_gpx(points, args.output, args.tz)
    print(f"Записано в {args.output}: {written} точек (после дедупликации)")

    tz = timezone(timedelta(hours=args.tz))
    days = sorted({dt.astimezone(tz).date() for _, _, dt in points})
    if days:
        print(f"Период: {days[0]} … {days[-1]} ({len(days)} дней)")


if __name__ == "__main__":
    main()