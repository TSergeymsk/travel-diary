#!/usr/bin/env python3
"""
Апгрейд config.yaml до актуальной схемы Travel Diary.

Что делает:
  • добавляет runtime.* и fallback.* значениями по умолчанию (если их нет);
  • добавляет processing.* новые ключи (дедуп, интерполяция GPS);
  • добавляет роль proofread (если её нет) на основе text с t=0.15;
  • сохраняет исходник как config.yaml.bak.YYYYMMDD-HHMMSS;
  • идемпотентен: повторный запуск ничего не портит.

Использование:
    python upgrade_config.py [path/to/config.yaml]
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml


# ---------- значения по умолчанию ----------

DEFAULTS_RUNTIME = {
    "ollama_host": "http://localhost:11434",
    "phase_mode": True,
    "preload": True,
    "unload_between_phases": True,
    "keep_alive": "30m",
}

DEFAULTS_FALLBACK = {
    "enabled": True,
    "reduce_role": "reduce",
    "hard_truncate_sentences": 3,
    "auto_reduce_over_chars": 0,
}

DEFAULTS_PROCESSING = {
    "vision_max_side": 768,
    "export_max_side": 1600,
    "cluster_radius_m": 400,
    "max_images_per_cluster": 4,
    "use_reverse_geocode": True,
    "max_photos_per_day": 6,
    "dedup_time_window_s": 5,
    "dedup_hash_threshold": 10,
    "max_photos_per_cluster": 2,
    "gps_interpolation_from_gpx": True,
    "gps_interpolation_max_gap_min": 30,
}

DEFAULTS_SEARCH = {
    "enabled": True,
    "language": "ru",
    "max_chars": 600,
    "timeout": 8,
}

DEFAULTS_OUTPUT = {
    "title": "Моё путешествие",
    "theme": "dark",
}


# ---------- утилиты ----------

def deep_merge_defaults(target: dict, defaults: dict) -> list[str]:
    """Добавляет в target ключи из defaults, которых там нет.
       Возвращает список имён добавленных ключей."""
    added = []
    for k, v in defaults.items():
        if k not in target:
            target[k] = v
            added.append(k)
    return added


def build_proofread_role(text_role: dict) -> dict:
    """
    Строит роль proofread на основе цепочки text.
    Берём только ollama-бэкенды (облачные не нужны — они дороже и не лучше для правки),
    temperature 0.15, max_tokens 1500.
    """
    chain = []
    for item in text_role.get("chain", []):
        if item.get("backend") != "ollama":
            continue
        new_item = dict(item)
        new_item["temperature"] = 0.15
        new_item["max_tokens"] = 1500
        # метка, чтобы в логах отличать корректора от генератора
        if "label" in new_item:
            new_item["label"] = new_item["label"] + "-proof"
        chain.append(new_item)

    return {
        "empty_result": "",
        "chain": chain or [{
            "backend": "ollama",
            "model": "qwen2.5:7b",
            "label": "local-qwen2.5-proof",
            "keep_alive": "30m",
            "temperature": 0.15,
            "max_tokens": 1500,
        }],
    }


# ---------- основная функция ----------

def upgrade(cfg: dict) -> tuple[dict, list[str]]:
    notes: list[str] = []

    # runtime
    runtime = cfg.setdefault("runtime", {})
    added = deep_merge_defaults(runtime, DEFAULTS_RUNTIME)
    if added:
        notes.append(f"runtime: добавлено {added}")

    # fallback
    fallback = cfg.setdefault("fallback", {})
    added = deep_merge_defaults(fallback, DEFAULTS_FALLBACK)
    if added:
        notes.append(f"fallback: добавлено {added}")

    # roles
    roles = cfg.setdefault("roles", {})
    if "vision" not in roles:
        notes.append("roles.vision отсутствует — пропускаю, поправьте руками")
    if "text" not in roles:
        notes.append("roles.text отсутствует — пропускаю, поправьте руками")

    if "proofread" not in roles and "text" in roles:
        roles["proofread"] = build_proofread_role(roles["text"])
        n = len(roles["proofread"]["chain"])
        notes.append(f"roles.proofread: добавлена ({n} шагов цепочки)")

    # если у vision/text нет fallback_role — выставим reduce
    for rn in ("vision", "text"):
        if rn in roles:
            r = roles[rn]
            if "fallback_role" not in r and "reduce" in roles:
                r["fallback_role"] = "reduce"
                notes.append(f"roles.{rn}.fallback_role = reduce")
            if "empty_result" not in r:
                r["empty_result"] = ("Кадры этого места" if rn == "vision" else "")
                notes.append(f"roles.{rn}.empty_result = {r['empty_result']!r}")

    # reduce: у него не должно быть fallback_role (защита от рекурсии)
    if "reduce" in roles:
        if "fallback_role" in roles["reduce"]:
            del roles["reduce"]["fallback_role"]
            notes.append("roles.reduce.fallback_role удалён (защита от рекурсии)")
        if "empty_result" not in roles["reduce"]:
            roles["reduce"]["empty_result"] = ""

    # processing
    processing = cfg.setdefault("processing", {})
    added = deep_merge_defaults(processing, DEFAULTS_PROCESSING)
    if added:
        notes.append(f"processing: добавлено {added}")

    # search / output
    search = cfg.setdefault("search", {})
    added = deep_merge_defaults(search, DEFAULTS_SEARCH)
    if added:
        notes.append(f"search: добавлено {added}")

    output = cfg.setdefault("output", {})
    added = deep_merge_defaults(output, DEFAULTS_OUTPUT)
    if added:
        notes.append(f"output: добавлено {added}")

    return cfg, notes


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.yaml")
    if not path.exists():
        sys.exit(f"Нет файла {path}")

    # бэкап
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{stamp}")
    shutil.copy2(path, backup)

    old = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    new, notes = upgrade(old)

    # ключи верхнего уровня в стабильном порядке
    top_order = ["runtime", "fallback", "roles",
                 "processing", "search", "output"]
    ordered: dict = {}
    for k in top_order:
        if k in new:
            ordered[k] = new[k]
    # на случай пользовательских секций — добавляем в конец
    for k in new:
        if k not in ordered:
            ordered[k] = new[k]

    path.write_text(
        yaml.safe_dump(ordered, allow_unicode=True,
                       sort_keys=False, width=100),
        encoding="utf-8",
    )

    print(f"Готово: {path}")
    print(f"Бэкап:  {backup}")
    if notes:
        print("\nИзменения:")
        for n in notes:
            print(f"  • {n}")
    else:
        print("\nИзменений не потребовалось — конфиг уже актуальный.")


if __name__ == "__main__":
    main()