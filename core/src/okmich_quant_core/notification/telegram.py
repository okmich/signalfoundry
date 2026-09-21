import html
from typing import Any, Dict

import requests

from .base import BaseNotifier, _AsyncDispatcher


class Telegram:
    """
    Low-level Telegram Bot API client. Makes a synchronous HTTP POST for each message.
    Use TelegramNotifier for non-blocking delivery inside a trading loop.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        :param config: dict with keys ``bot_token`` and ``chat_id``
        """
        self.bot_token = config["bot_token"]
        self.chat_id = config["chat_id"]

    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_notification": True,
        }
        # Explicit timeout: (connect_timeout, read_timeout) in seconds.
        # Without a timeout, requests.post can block the worker thread indefinitely on a slow or unreachable network.
        response = requests.post(
            url, json=payload, headers={"Content-Type": "application/json"},
            timeout=(5, 10),
        )
        response.raise_for_status()
        return True


class TelegramNotifier(BaseNotifier):
    """
    Non-blocking Telegram notifier. Every on_xxx() call enqueues a formatted HTML
    message and returns immediately; a daemon thread delivers it in the background.

    ``strategy_name`` names the sending system (pass ``SystemConfig.name``); it heads every message as a
    ``[tag]`` so alerts from many systems sharing one chat can be told apart.

    Usage::

        notifier = TelegramNotifier(bot_token="<token>", chat_id="<chat_id>", strategy_name=system_config.name)
        strategy = MyStrategy(config, signal, notifier=notifier)
    """

    def __init__(self, bot_token: str, chat_id: str, strategy_name: str = "", broker: str = ""):
        self._client = Telegram({"bot_token": bot_token, "chat_id": chat_id})
        self._dispatcher = _AsyncDispatcher(self._client.send_message)
        self.strategy_name = strategy_name
        self._broker = broker

    def _tags(self, *names: str) -> str:
        """`` [system] [name] ...``: the sending system first, then any per-message names, minus blanks and repeats.

        Every name is HTML-escaped: an unescaped '<', '>' or '&' makes Telegram reject the whole message.
        """
        unique = dict.fromkeys(name for name in (self.strategy_name, *names) if name)
        return "".join(f" [{html.escape(str(name))}]" for name in unique)

    # ------------------------------------------------------------------
    # BaseNotifier implementation
    # ------------------------------------------------------------------

    def on_trade_opened(self, symbol: str, direction: str, volume: float, price: float,
            sl: float, tp: float, magic: int, ticket: int):
        msg = \
            f"<b>🔵 OPENED</b>{self._tags(self._broker)} {html.escape(symbol)} {html.escape(direction)} " \
            f"{volume}L @ {price}\nSL: {sl} | TP: {tp} | #{ticket}"
        self._dispatcher.dispatch(msg)

    def on_trade_closed(self, symbol: str, ticket, profit: float, price: float = 0.0, reason: str = ""):
        head = "🟢 CLOSED" if profit > 0 else "🔴 CLOSED"
        why = f"  [{html.escape(str(reason))}]" if reason else ""
        at = f"  @ {price}" if price else ""
        msg = (f"<b>{head}</b>{self._tags(self._broker)} {html.escape(symbol)} #{html.escape(str(ticket))}{why}{at}"
               f"\nP&amp;L: {profit:+.2f}")
        self._dispatcher.dispatch(msg)

    def on_trade_modified(self, symbol: str, ticket: int, sl: float, tp: float):
        msg = f"<b>✏️ MODIFIED</b>{self._tags(self._broker)} {html.escape(symbol)} #{ticket}  SL→{sl}  TP→{tp}"
        self._dispatcher.dispatch(msg)

    def on_trade_failed(self, symbol: str, direction: str, reason: str, retcode: int = None, context: dict = None):
        # reason and direction may contain raw broker comments / exception
        # messages with '<', '>', '&'; HTML-escape so Telegram's HTML parser
        # doesn't reject the message and silently drop the alert.
        ctx = context or {}
        suffix = f" (retcode {retcode})" if retcode is not None else ""
        msg = \
            f"<b>🛑 TRADE FAILED</b>{self._tags(ctx.get('strategy_name', ''), self._broker)} " \
            f"{html.escape(symbol)} {html.escape(direction)}\n{html.escape(reason)}{suffix}"
        self._dispatcher.dispatch(msg)

    def on_error(self, strategy_name: str, error_message: str, context: dict = None):
        # error_message is usually str(exception), which often contains '<' (e.g. "'<' not supported between ...").
        msg = f"<b>⚠️ ERROR</b>{self._tags(strategy_name)}\n{html.escape(str(error_message))}"
        self._dispatcher.dispatch(msg)

    def on_circuit_breaker_tripped(self, strategy_name: str, consecutive_errors: int):
        msg = f"<b>🚫 CIRCUIT BREAKER</b>{self._tags(strategy_name)} tripped after {consecutive_errors} errors"
        self._dispatcher.dispatch(msg)

    def on_connection_lost(self, strategy_name: str):
        msg = f"<b>📡 CONNECTION LOST</b>{self._tags(strategy_name)}"
        self._dispatcher.dispatch(msg)

    def on_connection_restored(self, strategy_name: str):
        msg = f"<b>✅ CONNECTION RESTORED</b>{self._tags(strategy_name)}"
        self._dispatcher.dispatch(msg)

    def close(self):
        self._dispatcher.close()
