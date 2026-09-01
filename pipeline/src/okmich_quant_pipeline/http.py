"""Shared HTTP session for the pipeline's data fetchers (macro series, news calendar).

A single keep-alive session with a polite research User-Agent and a conservative timeout; no
fetcher should construct its own ``requests`` calls. ``get`` retries transient failures
(connection/timeout/5xx) with linear backoff and fails fast on client errors (4xx). Some
government sites (notably bls.gov) reject non-browser User-Agents with 403 — pass
``browser_ua=True`` for those; the header swap is scoped to the single request.
"""
from __future__ import annotations

import time

import requests

_USER_AGENT = "signalfoundry (research; contact: zwealthz5@gmail.com)"
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_DEFAULT_TIMEOUT_SECONDS = 30
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF_SECONDS = 1.5

_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Return the module-level requests session (lazy-initialised)."""
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": _USER_AGENT, "Accept": "*/*"})
        _session = s
    return _session


def get(url: str, *, timeout: int = _DEFAULT_TIMEOUT_SECONDS, retries: int = _DEFAULT_RETRIES,
        backoff_seconds: float = _DEFAULT_BACKOFF_SECONDS, browser_ua: bool = False) -> requests.Response:
    """GET ``url`` via the shared session and raise on non-2xx.

    Retries connection/timeout errors and 5xx responses up to ``retries`` times with linear
    backoff. 4xx responses raise immediately (retrying a bad request is pointless). Set
    ``browser_ua=True`` for sites that reject the research User-Agent (e.g. bls.gov) — the
    browser header is applied only to this request.
    """
    if retries < 1:
        raise ValueError(f"retries must be >= 1; got {retries}")
    headers = {"User-Agent": _BROWSER_UA} if browser_ua else None
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = get_session().get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and status < 500:
                raise  # client error — do not retry
            last_exc = exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        if attempt < retries:
            time.sleep(backoff_seconds * attempt)
    assert last_exc is not None  # only reached after at least one failed attempt
    raise last_exc
