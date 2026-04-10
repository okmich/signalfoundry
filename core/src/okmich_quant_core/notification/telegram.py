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

    Usage::

        notifier = TelegramNotifier(bot_token="<token>", chat_id="<chat_id>", strategy_name="MyStrategy")
        strategy = MyStrategy(config, signal, notifier=notifier)
    """

    def __init__(self, bot_token: str, chat_id: str, strategy_name: str = "", broker: str = ""):
        self._client = Telegram({"bot_token": bot_token, "chat_id": chat_id})
        self._dispatcher = _AsyncDispatcher(self._client.send_message)
        self.strategy_name = strategy_name
        self._broker_tag = f"[{broker}] " if broker else ""

    # ------------------------------------------------------------------
    # BaseNotifier implementation
    # ------------------------------------------------------------------

    def on_trade_opened(self, symbol: str, direction: str, volume: float, price: float,
            sl: float, tp: float, magic: int, ticket: int):
        msg = \
            f"<b>🟢 OPENED</b> {self._broker_tag}{symbol} {direction} {volume}L @ {price}\n" \
            f"SL: {sl} | TP: {tp} | #{ticket}"
        self._dispatcher.dispatch(msg)

    def on_trade_closed(self, symbol: str, ticket: int, profit: float):
        msg = f"<b>🔴 CLOSED</b> {self._broker_tag}{symbol} ticket #{ticket}\nP&amp;L: {profit}"
        self._dispatcher.dispatch(msg)

    def on_trade_modified(self, symbol: str, ticket: int, sl: float, tp: float):
        msg = f"<b>✏️ MODIFIED</b> {self._broker_tag}{symbol} #{ticket}  SL→{sl}  TP→{tp}"
        self._dispatcher.dispatch(msg)

    def on_error(self, strategy_name: str, error_message: str, context: dict = None):
        msg = f"<b>⚠️ ERROR</b> [{strategy_name}]\n{error_message}"
        self._dispatcher.dispatch(msg)

    def on_circuit_breaker_tripped(self, strategy_name: str, consecutive_errors: int):
        msg = \
            f"<b>🚫 CIRCUIT BREAKER</b> [{strategy_name}] " \
            f"tripped after {consecutive_errors} errors"
        self._dispatcher.dispatch(msg)

    def on_connection_lost(self, strategy_name: str):
        msg = f"<b>📡 CONNECTION LOST</b> [{strategy_name}]"
        self._dispatcher.dispatch(msg)

    def on_connection_restored(self, strategy_name: str):
        msg = f"<b>✅ CONNECTION RESTORED</b> [{strategy_name}]"
        self._dispatcher.dispatch(msg)

    def close(self):
        self._dispatcher.close()
