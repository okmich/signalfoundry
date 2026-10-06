"""Client order ids that encode the strategy's ``magic``.

Format: ``sf{magic}x{token}``. Only letters and digits, because that is the lowest common denominator across venues
(OKX accepts alphanumerics only; Bybit and Binance also allow ``-`` and ``_``). The ``x`` terminator after the magic
makes the prefix unambiguous: ``sf42x`` can never match an id of magic ``421``.

CCXT does not validate client order ids at all (a 64-char id with spaces is sent through untouched), so every id is
checked here against the venue profile's rule before it leaves the process.
"""
import re
import secrets
import time
from dataclasses import dataclass

_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_PREFIX_RE = re.compile(r"^sf(\d+)x")


@dataclass(frozen=True)
class ClientIdRule:
    """A venue's client order id constraint."""
    max_length: int
    pattern: str = r"^[A-Za-z0-9]+$"

    def check(self, client_id: str) -> None:
        if not client_id or len(client_id) > self.max_length or not re.match(self.pattern, client_id):
            raise ValueError(f"client order id {client_id!r} violates the venue rule "
                             f"(max {self.max_length} chars, pattern {self.pattern})")


def _to_base36(value: int) -> str:
    if value == 0:
        return "0"
    out = []
    while value:
        value, rem = divmod(value, 36)
        out.append(_BASE36[rem])
    return "".join(reversed(out))


def client_order_prefix(magic: int) -> str:
    if magic < 0:
        raise ValueError(f"magic must be non-negative to be encoded in a client order id (got {magic})")
    return f"sf{magic}x"


def make_client_order_id(magic: int, rule: ClientIdRule) -> str:
    """A fresh, unique client order id for ``magic`` that satisfies ``rule``.

    The token is a microsecond timestamp plus random characters, base36, truncated from the left of the random part
    only if the venue limit forces it. Uniqueness matters: a resend with the SAME id is how a timed-out placement is
    made idempotent, so two distinct orders must never share one.
    """
    prefix = client_order_prefix(magic)
    stamp = _to_base36(time.time_ns() // 1_000)
    noise = _to_base36(secrets.randbits(40))
    room = rule.max_length - len(prefix) - len(stamp)
    if room < 2:
        raise ValueError(f"magic {magic} leaves no room for a unique client order id within {rule.max_length} chars")
    client_id = prefix + stamp + noise[:room]
    rule.check(client_id)
    return client_id


def magic_of(client_id: str | None) -> int | None:
    """The magic encoded in one of our client order ids, or ``None`` for an id this package did not create."""
    if not client_id:
        return None
    match = _PREFIX_RE.match(client_id)
    return int(match.group(1)) if match else None


def is_ours(client_id: str | None, magic: int) -> bool:
    return magic_of(client_id) == magic
