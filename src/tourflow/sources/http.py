"""Транспорт для источников: безопасный HTTP-клиент и синтетический транспорт.

Безопасность и устойчивость:
* ходим только по https и только на хосты из allowlist в config.yaml (защита от SSRF);
* TLS-сертификаты проверяются всегда, есть таймауты и лимит размера ответа;
* повтор с экспоненциальной задержкой на сетевых ошибках, 5xx и 429 (с учётом Retry-After).
"""
from __future__ import annotations

import logging
import time
from typing import Protocol
from urllib.parse import urlparse

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from tourflow.config import AppConfig, get_settings

log = logging.getLogger(__name__)


class SourceUnavailable(RuntimeError):
    """Источник недоступен после всех повторов."""


class NotFound(RuntimeError):
    """Источник ответил 404 — данных за период нет (это не сбой)."""


class Transport(Protocol):
    def get(self, url: str, params: dict | None = None) -> tuple[bytes, str]: ...


class _Retryable(Exception):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (_Retryable, httpx.TransportError))


def _wait(state: RetryCallState) -> float:
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, _Retryable) and exc.retry_after:
        return min(exc.retry_after, 120.0)
    return wait_exponential_jitter(initial=1, max=60)(state)


class HttpTransport:
    def __init__(self, cfg: AppConfig):
        h = cfg.sources.http
        self.allowed = set(cfg.sources.allowed_hosts)
        self.max_bytes = h.max_response_mb * 1024 * 1024
        self.max_retries = h.max_retries
        contact = get_settings().contact.strip()
        if not contact:
            log.warning("TOURFLOW_CONTACT не задан: Wikimedia может отклонять запросы (403)")
        ua = h.user_agent.format(contact=contact or "contact not set")
        self.client = httpx.Client(
            timeout=httpx.Timeout(h.timeout_s),
            headers={"User-Agent": ua},
            follow_redirects=True,
            max_redirects=3,
            verify=True,
            # каждый запрос, включая редиректы, проверяется по allowlist
            event_hooks={"request": [lambda req: self._check_url(str(req.url))]},
        )

    def _check_url(self, url: str) -> None:
        u = urlparse(url)
        if u.scheme != "https":
            raise ValueError(f"разрешён только https: {url}")
        if u.hostname not in self.allowed:
            raise ValueError(f"хост {u.hostname} не в allowlist (config.yaml: sources.allowed_hosts)")

    def get(self, url: str, params: dict | None = None) -> tuple[bytes, str]:
        self._check_url(url)

        @retry(retry=retry_if_exception(_is_retryable), wait=_wait,
               stop=stop_after_attempt(self.max_retries), reraise=True)
        def _do() -> tuple[bytes, str]:
            with self.client.stream("GET", url, params=params) as r:
                if r.status_code == 404:
                    raise NotFound(url)
                if r.status_code == 429 or r.status_code >= 500:
                    ra = r.headers.get("Retry-After")
                    raise _Retryable(f"HTTP {r.status_code}", float(ra) if ra and ra.isdigit() else None)
                r.raise_for_status()
                buf = bytearray()
                for chunk in r.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > self.max_bytes:
                        raise ValueError(f"ответ больше {self.max_bytes} байт: {url}")
                return bytes(buf), r.headers.get("Content-Type", "application/octet-stream")

        try:
            return _do()
        except (_Retryable, httpx.TransportError) as exc:
            raise SourceUnavailable(f"{url}: {exc}") from exc


class SyntheticTransport:
    """Отвечает так же, как настоящие API, но данными синтетического мира. Сеть не нужна."""

    def __init__(self, cfg: AppConfig, world):
        self.cfg = cfg
        self.world = world
        self.calls = 0

    def get(self, url: str, params: dict | None = None) -> tuple[bytes, str]:
        self.calls += 1
        time.sleep(0)  # точка для имитации задержек в тестах
        return self.world.respond(url, params or {})
