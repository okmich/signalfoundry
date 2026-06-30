"""Shared HTTP session for the macro fetchers.

Mirrors ``utilities.news_calendar.http`` but kept self-contained — the macro package has no dependency on the calendar package. A single keep-alive session with a polite research
User-Agent and conservative timeout; no fetcher should construct its own ``requests`` calls.
``get`` retries transient failures (connection/timeout/5xx) with linear backoff; client errors
(4xx, e.g. a bad series id) fail fast.
"""
from __future__ import annotations

import time

import requests

_USER_AGENT = "signalfoundry-lab/macro (research; contact: zwealthz5@gmail.com)"
_DEFAULT_TIMEOUT_SECONDS = 30
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF_SECONDS = 1.5

_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Return the module-level requests session (lazy-initialised)."""
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": _USER_AGENT, "Accept": "text/csv,text/plain,*/*"})
        _session = s
    return _session


def get(url: str, *, timeout: int = _DEFAULT_TIMEOUT_SECONDS, retries: int = _DEFAULT_RETRIES,
        backoff_seconds: float = _DEFAULT_BACKOFF_SECONDS) -> requests.Response:
    """GET ``url`` via the shared session and raise on non-2xx.

    Retries connection/timeout errors and 5xx responses up to ``retries`` times with linear
    backoff. 4xx responses raise immediately (retrying a bad request is pointless).
    """
    if retries < 1:
        raise ValueError(f"retries must be >= 1; got {retries}")
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = get_session().get(url, timeout=timeout)
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
