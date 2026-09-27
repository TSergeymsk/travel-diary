"""
Провайдеры LLM: единый интерфейс к Ollama, Groq, OpenRouter.

- BackendConfig    — декларативное описание одного бэкенда
- RateLimiter      — sliding-window RPM/TPM-ограничитель
- make_backend()   — фабрика клиентов с retry и rate limit
- RoleRunner       — цепочка с fallback на другую роль и hard-truncate
- Registry         — кэш бэкендов и ролей, управление фазами VRAM

Облачные бэкенды (Groq, OpenRouter) оборачиваются в:
  • RateLimiter — не даём выйти за RPM/TPM;
  • retry с экспоненциальным backoff — 429 и 5xx переживаем без падения.

reasoning_effort: low | medium | high — для reasoning-моделей
(например gpt-oss-120b/20b на Groq). Без него модель тратит весь
бюджет max_tokens на внутренние «размышления», а content приходит пустым.
"""
from __future__ import annotations

import base64
import logging
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

log = logging.getLogger("providers")


# ============================================================
#  Конфиг одного бэкенда
# ============================================================

@dataclass
class RateLimitConfig:
    """Sliding-window лимиты для облачных API (RPM/TPM за 60 сек)."""
    rpm: int = 0
    tpm: int = 0
    min_interval_s: float = 0.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "RateLimitConfig":
        if not d:
            return cls()
        return cls(
            rpm=int(d.get("rpm", 0)),
            tpm=int(d.get("tpm", 0)),
            min_interval_s=float(d.get("min_interval_s", 0.0)),
        )


@dataclass
class RetryConfig:
    max_attempts: int = 4
    base_backoff_s: float = 2.0
    max_backoff_s: float = 45.0
    jitter_s: float = 1.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "RetryConfig":
        if not d:
            return cls()
        return cls(
            max_attempts=int(d.get("max_attempts", 4)),
            base_backoff_s=float(d.get("base_backoff_s", 2.0)),
            max_backoff_s=float(d.get("max_backoff_s", 45.0)),
            jitter_s=float(d.get("jitter_s", 1.0)),
        )


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
    num_ctx: int | None = None
    reasoning_effort: str | None = None       # low | medium | high
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "BackendConfig":
        rl = RateLimitConfig.from_dict(d.get("rate_limit"))
        kwargs = {k: d[k] for k in cls.__dataclass_fields__
                  if k in d and k != "rate_limit"}
        return cls(rate_limit=rl, **kwargs)

    @property
    def name(self) -> str:
        return self.label or f"{self.backend}:{self.model}"


# ============================================================
#  Rate limiter
# ============================================================

class RateLimiter:
    """
    Sliding-window: не более rpm запросов и tpm токенов за последние 60 сек.
    Плюс минимальный интервал между запросами.
    """

    def __init__(self, rpm: int = 0, tpm: int = 0, min_interval_s: float = 0.0):
        self.rpm = rpm
        self.tpm = tpm
        self.min_interval = min_interval_s
        self._calls: deque[tuple[float, int]] = deque()
        self._last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int = 0) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                self._purge(now)
                wait = 0.0

                if self.rpm > 0 and len(self._calls) >= self.rpm:
                    oldest = self._calls[0][0]
                    wait = max(wait, 60.0 - (now - oldest) + 0.2)

                if self.tpm > 0 and estimated_tokens > 0 and self._calls:
                    total = sum(t for _, t in self._calls)
                    if total + estimated_tokens > self.tpm:
                        oldest = self._calls[0][0]
                        wait = max(wait, 60.0 - (now - oldest) + 0.2)

                if self.min_interval > 0:
                    elapsed = now - self._last_call
                    if elapsed < self.min_interval:
                        wait = max(wait, self.min_interval - elapsed)

                if wait <= 0:
                    break
                time.sleep(min(wait, 30.0))

            now = time.monotonic()
            self._calls.append((now, estimated_tokens))
            self._last_call = now

    def pause(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)
            with self._lock:
                self._last_call = time.monotonic()

    def _purge(self, now: float) -> None:
        cutoff = now - 60.0
        while self._calls and self._calls[0][0] < cutoff:
            self._calls.popleft()

    def tokens_in_window(self) -> int:
        with self._lock:
            self._purge(time.monotonic())
            return sum(t for _, t in self._calls)


def estimate_tokens(text: str, max_output: int) -> int:
    """Грубая оценка: русский текст ≈ 1 токен на 3 символа."""
    return max(1, len(text) // 3) + max_output


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
        return {"keep_alive": self.cfg.keep_alive} if self.cfg.keep_alive else {}

    def _options(self) -> dict:
        opts = {
            "temperature": self.cfg.temperature,
            "num_predict": self.cfg.max_tokens,
        }
        if self.cfg.num_ctx:
            opts["num_ctx"] = self.cfg.num_ctx
        return opts

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
    """
    Облачный бэкенд с rate limiting и retry.
    Подходит для Groq, OpenRouter и любого OpenAI-совместимого API.
    """

    def __init__(self, cfg: BackendConfig, retry_cfg: RetryConfig):
        from openai import OpenAI
        if not cfg.api_key_env:
            raise ValueError(f"{cfg.name}: не задан api_key_env")
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{cfg.name}: переменная {cfg.api_key_env} пуста")

        self.cfg = cfg
        self.retry_cfg = retry_cfg
        self.client = OpenAI(api_key=key, base_url=cfg.base_url)
        self.rate_limiter = RateLimiter(
            rpm=cfg.rate_limit.rpm,
            tpm=cfg.rate_limit.tpm,
            min_interval_s=cfg.rate_limit.min_interval_s,
        )

    # ---------- извлечение статуса из openai-ошибок ----------

    @staticmethod
    def _status(e: Exception) -> int | None:
        st = getattr(e, "status_code", None)
        if st is None:
            resp = getattr(e, "response", None)
            if resp is not None:
                st = getattr(resp, "status_code", None)
        return st

    @staticmethod
    def _retry_after(e: Exception) -> float | None:
        resp = getattr(e, "response", None)
        if resp is None:
            return None
        headers = getattr(resp, "headers", None)
        if not headers:
            return None
        for key in ("retry-after", "x-ratelimit-reset-requests",
                    "x-ratelimit-reset-tokens"):
            val = headers.get(key)
            if not val:
                continue
            try:
                return float(str(val).rstrip("s"))
            except (TypeError, ValueError):
                continue
        return None

    # ---------- вызов с retry и rate limit ----------

    def _call_with_retry(self, fn: Callable[[], Any], estimated_tokens: int):
        last_exc: Exception | None = None
        for attempt in range(self.retry_cfg.max_attempts):
            self.rate_limiter.acquire(estimated_tokens)
            try:
                return fn()
            except Exception as e:
                last_exc = e
                status = self._status(e)

                if status == 429:
                    ra = self._retry_after(e)
                    if ra and ra > 0:
                        wait = min(ra, self.retry_cfg.max_backoff_s)
                    else:
                        base = self.retry_cfg.base_backoff_s * (2 ** attempt)
                        wait = min(base, self.retry_cfg.max_backoff_s)
                    wait += random.uniform(0, self.retry_cfg.jitter_s)
                    log.warning(
                        f"{self.cfg.name}: 429, sleep {wait:.1f}s "
                        f"(attempt {attempt + 1}/{self.retry_cfg.max_attempts})"
                    )
                    self.rate_limiter.pause(wait)
                    continue

                if status is not None and 500 <= status < 600:
                    base = self.retry_cfg.base_backoff_s * (2 ** attempt)
                    wait = min(base, self.retry_cfg.max_backoff_s)
                    wait += random.uniform(0, self.retry_cfg.jitter_s)
                    log.warning(
                        f"{self.cfg.name}: {status}, retry in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue

                raise

        assert last_exc is not None
        raise last_exc

    # ---------- публичные методы ----------

    def complete(self, system: str, user: str) -> str:
        prompt_text = system + "\n" + user
        estimated = estimate_tokens(prompt_text, self.cfg.max_tokens)

        def _do():
            kwargs: dict = {
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            }
            if self.cfg.reasoning_effort:
                kwargs["reasoning_effort"] = self.cfg.reasoning_effort
            return self.client.chat.completions.create(**kwargs)

        r = self._call_with_retry(_do, estimated)
        return (r.choices[0].message.content or "").strip()

    def describe_images(self, images: list[bytes], prompt: str) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        estimated = estimate_tokens(prompt, self.cfg.max_tokens) + len(images) * 300

        def _do():
            kwargs: dict = {
                "model": self.cfg.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            }
            if self.cfg.reasoning_effort:
                kwargs["reasoning_effort"] = self.cfg.reasoning_effort
            return self.client.chat.completions.create(**kwargs)

        r = self._call_with_retry(_do, estimated)
        return (r.choices[0].message.content or "").strip()


def make_backend(
    cfg: BackendConfig,
    default_ollama_host: str = "http://localhost:11434",
    retry_cfg: RetryConfig | None = None,
):
    if cfg.backend == "ollama":
        return _OllamaBackend(cfg, default_ollama_host)
    if cfg.backend in ("groq", "openrouter", "openai"):
        return _OpenAIBackend(cfg, retry_cfg or RetryConfig())
    raise ValueError(f"Неизвестный backend: {cfg.backend}")


# ============================================================
#  Управление VRAM
# ============================================================

class OllamaManager:
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

    def loaded(self) -> list[dict]:
        try:
            r = httpx.get(f"{self.host}/api/ps", timeout=5)
            return r.json().get("models", [])
        except Exception:
            return []

    def loaded_names(self) -> list[str]:
        return [m["name"] for m in self.loaded()]

    def unload_all(self) -> None:
        for m in self.loaded_names():
            self.unload(m)

    def ping(self) -> bool:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def vram_report(self) -> str:
        items = self.loaded()
        if not items:
            return "(VRAM пуст)"
        lines = []
        for m in items:
            size_gb = m.get("size", 0) / 1e9
            vram_gb = m.get("size_vram", 0) / 1e9
            pct = 100 * vram_gb / size_gb if size_gb else 0
            lines.append(f"  {m['name']}: {vram_gb:.2f} / {size_gb:.2f} GB "
                         f"в VRAM ({pct:.0f}%)")
        return "\n".join(lines)


@dataclass
class Phase:
    name: str
    ollama_models: set[str]
    t0: float = 0.0


@dataclass
class Attempt:
    backend_name: str
    ok: bool
    error: str | None = None
    duration_s: float = 0.0


# ============================================================
#  RoleRunner
# ============================================================

class RoleRunner:
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

    def _try_each(self, call: Callable[[Any], str | None]) -> str | None:
        if not self.chain:
            return None
        for cfg in self.chain:
            t0 = time.monotonic()
            try:
                backend = self.registry.get(cfg)
                result = call(backend)
                dt = time.monotonic() - t0
                # пустой результат считаем провалом — частая беда reasoning-моделей
                if result is None or (isinstance(result, str) and not result.strip()):
                    self.history.append(Attempt(cfg.name, False,
                                                "empty result", dt))
                    log.warning(f"[{self.role_name}] {cfg.name} empty result ({dt:.1f}s)")
                    continue
                self.history.append(Attempt(cfg.name, True, duration_s=dt))
                log.info(f"[{self.role_name}] {cfg.name} ok ({dt:.1f}s)")
                return result
            except Exception as e:
                dt = time.monotonic() - t0
                self.history.append(Attempt(cfg.name, False, str(e), dt))
                log.warning(f"[{self.role_name}] {cfg.name} failed ({dt:.1f}s): {e}")
                continue
        return None

    def text(self, system: str, user: str, reduce_source: str = "") -> str:
        result = self._try_each(lambda b: b.complete(system, user))
        if result:
            if self.fb_auto_reduce_over and len(result) > self.fb_auto_reduce_over:
                log.info(f"[{self.role_name}] auto-reduce: {len(result)} chars")
                return self._reduce(result)
            return result
        log.warning(f"[{self.role_name}] все бэкенды исчерпаны, fallback")
        if reduce_source:
            return self._reduce(reduce_source)
        return self.empty_result

    def vision(self, images: list[bytes], prompt: str) -> str:
        if not images:
            return self.empty_result
        result = self._try_each(lambda b: b.describe_images(images, prompt))
        return result or self.empty_result

    def _reduce(self, source: str) -> str:
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

        self.ollama_host = (
            self.runtime.get("ollama_host")
            or self._host_from_chain(config)
            or "http://localhost:11434"
        )
        self.ollama = OllamaManager(self.ollama_host)
        self.retry_cfg = RetryConfig.from_dict(self.runtime.get("retry"))

    @staticmethod
    def _host_from_chain(config: dict) -> str | None:
        for role_cfg in config.get("roles", {}).values():
            for c in role_cfg.get("chain", []):
                if c.get("backend") == "ollama" and c.get("host"):
                    return c["host"]
        return None

    def get(self, cfg: BackendConfig):
        if cfg.name not in self._backends:
            self._backends[cfg.name] = make_backend(
                cfg, self.ollama_host, self.retry_cfg)
        return self._backends[cfg.name]

    def role(self, name: str) -> RoleRunner:
        if name not in self._roles:
            if name not in self.config["roles"]:
                raise KeyError(f"Роль '{name}' не описана в config.yaml")
            self._roles[name] = RoleRunner(name, self.config["roles"][name], self)
        return self._roles[name]

    def has_role(self, name: str) -> bool:
        return name in self.config.get("roles", {})

    def begin_phase(self, *role_names: str) -> Phase:
        if not role_names:
            raise ValueError("begin_phase: не передано ни одной роли")

        wanted: set[str] = set()
        first_model: str | None = None
        for rn in role_names:
            runner = self.role(rn)
            for c in runner.chain:
                if c.backend == "ollama":
                    wanted.add(c.model)
                    if first_model is None:
                        first_model = c.model

        label = "+".join(role_names)
        phase = Phase(name=label, ollama_models=wanted, t0=time.monotonic())

        if not self.runtime.get("phase_mode", True):
            return phase

        print(f"[phase] enter '{label}' @ {self.ollama_host}, "
              f"models: {wanted or '— (только облако)'}")

        if wanted and not self.ollama.ping():
            print(f"[phase]   ⚠ {self.ollama_host} недоступен — "
                  f"ollama-бэкенды упадут, сработает fallback")

        if self.runtime.get("unload_between_phases", True):
            for m in self.ollama.loaded_names():
                if m not in wanted:
                    print(f"[phase]   unloading {m}")
                    self.ollama.unload(m)

        if wanted and self.runtime.get("preload", True) and first_model:
            print(f"[phase]   preloading {first_model}")
            self.ollama.preload(first_model, self.runtime.get("keep_alive", "30m"))
            print(f"[phase]   VRAM:\n{self.ollama.vram_report()}")

        return phase

    def end_phase(self, phase: Phase) -> None:
        dt = time.monotonic() - phase.t0
        print(f"[phase] exit '{phase.name}' ({dt:.1f}s)")
        if not self.runtime.get("phase_mode", True):
            return
        if self.runtime.get("unload_between_phases", True):
            for m in phase.ollama_models:
                self.ollama.unload(m)

    def report(self) -> str:
        lines = [f"Ollama host: {self.ollama_host}", "Отчёт по ролям:"]
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