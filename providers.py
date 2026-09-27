"""
Провайдеры LLM: единый интерфейс к Ollama, Groq, OpenRouter.

- BackendConfig    — декларативное описание одного бэкенда
- make_backend()   — фабрика клиентов
- RoleRunner       — цепочка с fallback на другую роль и hard-truncate
- Registry         — кэш бэкендов и ролей, управление фазами VRAM

Поддерживает удалённый Ollama: адрес задаётся глобально через
runtime.ollama_host в config.yaml и наследуется всеми ollama-бэкендами,
если у конкретного бэкенда не указан собственный host.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

log = logging.getLogger("providers")


# ============================================================
#  Конфиг одного бэкенда
# ============================================================

@dataclass
class BackendConfig:
    backend: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    host: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    keep_alive: str | None = None
    label: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "BackendConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @property
    def name(self) -> str:
        return self.label or f"{self.backend}:{self.model}"


# ============================================================
#  Бэкенды
# ============================================================

class _OllamaBackend:
    def __init__(self, cfg: BackendConfig, default_host: str = "http://localhost:11434"):
        import ollama
        self.cfg = cfg
        host = cfg.host or default_host
        self.client = ollama.Client(host=host)

    def _kwargs(self) -> dict:
        if self.cfg.keep_alive:
            return {"keep_alive": self.cfg.keep_alive}
        return {}

    def _options(self) -> dict:
        return {
            "temperature": self.cfg.temperature,
            "num_predict": self.cfg.max_tokens,
        }

    def complete(self, system: str, user: str) -> str:
        r = self.client.chat(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options=self._options(),
            **self._kwargs(),
        )
        return (r.message.content or "").strip()

    def describe_images(self, images: list[bytes], prompt: str) -> str:
        r = self.client.chat(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt, "images": images}],
            options=self._options(),
            **self._kwargs(),
        )
        return (r.message.content or "").strip()


class _OpenAIBackend:
    def __init__(self, cfg: BackendConfig):
        from openai import OpenAI
        if not cfg.api_key_env:
            raise ValueError(f"{cfg.name}: не задан api_key_env")
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{cfg.name}: переменная {cfg.api_key_env} пуста")
        self.cfg = cfg
        self.client = OpenAI(api_key=key, base_url=cfg.base_url)

    def complete(self, system: str, user: str) -> str:
        r = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return (r.choices[0].message.content or "").strip()

    def describe_images(self, images: list[bytes], prompt: str) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        r = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return (r.choices[0].message.content or "").strip()


def make_backend(cfg: BackendConfig, default_ollama_host: str = "http://localhost:11434"):
    if cfg.backend == "ollama":
        return _OllamaBackend(cfg, default_ollama_host)
    if cfg.backend in ("groq", "openrouter", "openai"):
        return _OpenAIBackend(cfg)
    raise ValueError(f"Неизвестный backend: {cfg.backend}")


# ============================================================
#  Управление VRAM
# ============================================================

class OllamaManager:
    """Загрузка/выгрузка моделей Ollama из VRAM. Работает и с удалённым сервером."""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip("/")

    def preload(self, model: str, keep_alive: str = "30m") -> bool:
        try:
            r = httpx.post(
                f"{self.host}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": keep_alive},
                timeout=300,
            )
            return r.status_code == 200
        except Exception as e:
            log.warning(f"preload {model} @ {self.host}: {e}")
            return False

    def unload(self, model: str) -> None:
        try:
            httpx.post(
                f"{self.host}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
                timeout=30,
            )
        except Exception as e:
            log.warning(f"unload {model} @ {self.host}: {e}")

    def loaded(self) -> list[str]:
        try:
            r = httpx.get(f"{self.host}/api/ps", timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def unload_all(self) -> None:
        for m in self.loaded():
            self.unload(m)

    def ping(self) -> bool:
        """Проверка доступности сервера."""
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


@dataclass
class Phase:
    name: str
    ollama_models: set[str]
    t0: float = 0.0


# ============================================================
#  Attempt history
# ============================================================

@dataclass
class Attempt:
    backend_name: str
    ok: bool
    error: str | None = None
    duration_s: float = 0.0


# ============================================================
#  RoleRunner — цепочка с fallback
# ============================================================

class RoleRunner:
    """
    Одна роль (vision/text/reduce). Хранит цепочку бэкендов,
    на ошибку пробует следующий. Если вся цепочка упала —
    обращается к reduce-роли (имя из fallback_role), затем
    к hard-truncate, затем к empty_result.
    """

    def __init__(self, role_name: str, cfg: dict, registry: "Registry"):
        self.role_name = role_name
        self.registry = registry
        self.chain = [BackendConfig.from_dict(c) for c in cfg.get("chain", [])]
        self.fallback_role: str | None = cfg.get("fallback_role")
        self.empty_result: str = cfg.get("empty_result", "")
        self.history: list[Attempt] = []

        fb = registry.config.get("fallback", {})
        self.fb_enabled: bool = bool(fb.get("enabled", True))
        self.fb_truncate_n: int = int(fb.get("hard_truncate_sentences", 3))
        self.fb_auto_reduce_over: int = int(fb.get("auto_reduce_over_chars", 0))

    # ---------- внутренний обход цепочки ----------

    def _try_each(self, call: Callable[[Any], str | None]) -> str | None:
        if not self.chain:
            return None
        for cfg in self.chain:
            t0 = time.monotonic()
            try:
                backend = self.registry.get(cfg)
                result = call(backend)
                dt = time.monotonic() - t0
                self.history.append(Attempt(cfg.name, True, duration_s=dt))
                log.info(f"[{self.role_name}] {cfg.name} ok ({dt:.1f}s)")
                return result
            except Exception as e:
                dt = time.monotonic() - t0
                self.history.append(Attempt(cfg.name, False, str(e), dt))
                log.warning(f"[{self.role_name}] {cfg.name} failed ({dt:.1f}s): {e}")
                continue
        return None

    # ---------- публичные методы ----------

    def text(self, system: str, user: str, reduce_source: str = "") -> str:
        result = self._try_each(lambda b: b.complete(system, user))

        if result:
            if self.fb_auto_reduce_over and len(result) > self.fb_auto_reduce_over:
                log.info(f"[{self.role_name}] auto-reduce: {len(result)} chars")
                return self._reduce(result)
            return result

        log.error(f"[{self.role_name}] вся цепочка упала")
        if reduce_source:
            return self._reduce(reduce_source)
        return self.empty_result

    def vision(self, images: list[bytes], prompt: str) -> str:
        if not images:
            return self.empty_result
        result = self._try_each(lambda b: b.describe_images(images, prompt))
        return result or self.empty_result

    # ---------- аварийное сжатие ----------

    def _reduce(self, source: str) -> str:
        """Сжать текст через reduce-роль. При провале — hard-truncate."""
        source = (source or "").strip()
        if not source:
            return self.empty_result

        if not self.fb_enabled or not self.fallback_role:
            return self._hard_truncate(source)

        if self.fallback_role == self.role_name:
            log.warning(f"[{self.role_name}] fallback_role указывает на себя — truncate")
            return self._hard_truncate(source)

        reduce_role = self.registry.role(self.fallback_role)
        sys_prompt = (
            "Сожми текст до 2–3 предложений, сохранив ключевые факты. "
            "Без вступлений, без пояснений, только результат."
        )
        result = reduce_role._try_each(
            lambda b: b.complete(sys_prompt, source[:4000])
        )
        if result:
            return result
        return self._hard_truncate(source)

    def _hard_truncate(self, source: str) -> str:
        sentences = [s.strip() for s in (source or "").split(".") if s.strip()]
        if not sentences:
            return self.empty_result
        return ". ".join(sentences[: self.fb_truncate_n]) + "."


# ============================================================
#  Registry
# ============================================================

class Registry:
    def __init__(self, config: dict):
        self.config = config
        self.runtime = config.get("runtime", {})
        self._backends: dict[str, Any] = {}
        self._roles: dict[str, RoleRunner] = {}

        # единый host для всего проекта
        self.ollama_host = (
            self.runtime.get("ollama_host")
            or self._host_from_chain(config)
            or "http://localhost:11434"
        )
        self.ollama = OllamaManager(self.ollama_host)

    @staticmethod
    def _host_from_chain(config: dict) -> str | None:
        """Совместимость: если ollama_host не задан, ищем явный host в бэкендах."""
        for role_cfg in config.get("roles", {}).values():
            for c in role_cfg.get("chain", []):
                if c.get("backend") == "ollama" and c.get("host"):
                    return c["host"]
        return None

    def get(self, cfg: BackendConfig):
        if cfg.name not in self._backends:
            self._backends[cfg.name] = make_backend(cfg, self.ollama_host)
        return self._backends[cfg.name]

    def role(self, name: str) -> RoleRunner:
        if name not in self._roles:
            if name not in self.config["roles"]:
                raise KeyError(f"Роль '{name}' не описана в config.yaml")
            self._roles[name] = RoleRunner(name, self.config["roles"][name], self)
        return self._roles[name]

    # ---------- фазы VRAM ----------

    def begin_phase(self, role_name: str) -> Phase:
        runner = self.role(role_name)
        wanted = {c.model for c in runner.chain if c.backend == "ollama"}
        phase = Phase(name=role_name, ollama_models=wanted, t0=time.monotonic())

        if not self.runtime.get("phase_mode", True):
            return phase

        print(f"[phase] enter '{role_name}' @ {self.ollama_host}, "
              f"models: {wanted or '—'}")

        # проверка доступности сервера
        if not self.ollama.ping():
            print(f"[phase]   ⚠ {self.ollama_host} недоступен — "
                  f"ollama-бэкенды упадут, сработает fallback")

        if self.runtime.get("unload_between_phases", True):
            for m in self.ollama.loaded():
                if m not in wanted:
                    print(f"[phase]   unloading {m}")
                    self.ollama.unload(m)
        if self.runtime.get("preload", True) and wanted:
            first = next(c.model for c in runner.chain if c.backend == "ollama")
            print(f"[phase]   preloading {first}")
            self.ollama.preload(first, self.runtime.get("keep_alive", "30m"))
        return phase

    def end_phase(self, phase: Phase) -> None:
        dt = time.monotonic() - phase.t0
        print(f"[phase] exit '{phase.name}' ({dt:.1f}s)")
        if not self.runtime.get("phase_mode", True):
            return
        if self.runtime.get("unload_between_phases", True):
            for m in phase.ollama_models:
                self.ollama.unload(m)

    # ---------- отчёт ----------

    def report(self) -> str:
        lines = [f"Ollama host: {self.ollama_host}"]
        lines.append("Отчёт по ролям:")
        for name, role in self._roles.items():
            ok = sum(1 for a in role.history if a.ok)
            fail = sum(1 for a in role.history if not a.ok)
            if not ok and not fail:
                continue
            lines.append(f"  {name}: {ok} ok / {fail} fail")
            for a in role.history:
                if not a.ok:
                    lines.append(f"    ✗ {a.backend_name}: {a.error}")
        return "\n".join(lines)