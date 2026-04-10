import logging

from .base import BaseMt5PositionManager
from okmich_quant_core import StrategyConfig


class BasePercentBasedPositionManager(BaseMt5PositionManager):
    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        config = strategy_config.position_manager
        self.sl_percent = config.sl if config.sl else 0.0
        self.tp_percent = config.tp if config.tp else 0.0
        self.trailing_percent = config.trailing
        self.break_even_percent = config.break_even

    def _set_initial_sl_tp_percent(self, position, is_long: bool) -> tuple[float, float]:
        """
        Set initial SL/TP on the broker if not already present.

        Returns (effective_sl, effective_tp) — the values that are now (or were
        already) on the broker, so callers can use them immediately without
        waiting for a position refresh next cycle.  On modification failure the
        original (stale) values are returned so the caller can decide whether to
        continue.
        """
        ticket = position["ticket"]
        sl = position["sl"]
        tp = position["tp"]
        price_open = position["price_open"]

        if sl != 0 and (tp != 0 or self.tp_percent == 0):
            return sl, tp

        new_sl = 0.0
        new_tp = 0.0

        if sl == 0 and self.sl_percent > 0:
            if is_long:
                new_sl = price_open * (1 - self.sl_percent / 100)
            else:
                new_sl = price_open * (1 + self.sl_percent / 100)

        if tp == 0 and self.tp_percent > 0:
            if is_long:
                new_tp = price_open * (1 + self.tp_percent / 100)
            else:
                new_tp = price_open * (1 - self.tp_percent / 100)

        if new_sl <= 0 and sl == 0 and self.sl_percent > 0:
            logging.error(f"Computed SL <= 0 for {self.symbol} position {ticket}: {new_sl}. SL will not be set.")
            new_sl = 0.0
        if new_tp <= 0 and tp == 0 and self.tp_percent > 0:
            logging.error(f"Computed TP <= 0 for {self.symbol} position {ticket}: {new_tp}. TP will not be set.")
            new_tp = 0.0

        # Only call modify_position if we actually need to set SL or TP
        if new_sl > 0 or new_tp > 0:
            if self.modify_position(position, sl=new_sl if new_sl > 0 else None, tp=new_tp if new_tp > 0 else None):
                logging.info(
                    f"Set initial SL/TP for {self.symbol} ({self.magic}) position {ticket}. SL: {new_sl}, TP: {new_tp}"
                )
                return new_sl if new_sl > 0 else sl, new_tp if new_tp > 0 else tp
            else:
                logging.error(
                    f"Failed to set initial SL/TP for {self.symbol} ({self.magic}) position {ticket}. "
                    f"SL: {new_sl}, TP: {new_tp}"
                )
                return sl, tp  # modification failed; return original stale values
        return sl, tp  # nothing needed changing


class FixedPercentBasedPositionManager(BasePercentBasedPositionManager):
    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)

    def manage_long_position(self, position, flag: bool):
        if flag:
            self._manage_position(position, is_long=True)

    def manage_short_position(self, position, flag: bool):
        if flag:
            self._manage_position(position, is_long=False)

    def _manage_position(self, position, is_long: bool):
        self._set_initial_sl_tp_percent(position, is_long)


class FixedPercentSlTpWithTrailingPercentBasedPositionManager(BasePercentBasedPositionManager):
    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        if self.trailing_percent is None:
            raise ValueError(f"{type(self).__name__} requires 'trailing' in position_manager config")

    def manage_long_position(self, position, flag: bool):
        if flag:
            self._manage_position(position, is_long=True)

    def manage_short_position(self, position, flag: bool):
        if flag:
            self._manage_position(position, is_long=False)

    def _manage_position(self, position, is_long: bool):
        ticket = position["ticket"]
        price_open = position["price_open"]
        price_current = position["price_current"]

        # Set initial SL/TP if they don't exist; use returned effective values so trailing checks in this same cycle use
        # the actual broker sl, not the stale sl=0 snapshot from the position dict.
        sl, tp = self._set_initial_sl_tp_percent(position, is_long)

        if is_long and price_current > price_open:
            trailing_amount = price_current * (self.trailing_percent / 100)
            new_sl = price_current - trailing_amount
            if new_sl > sl:
                if self.modify_position(position, sl=new_sl):
                    logging.info(f"Trailed SL for long position {ticket} to {new_sl}")
        elif not is_long and price_current < price_open:
            trailing_amount = price_current * (self.trailing_percent / 100)
            new_sl = price_current + trailing_amount
            if new_sl < sl:
                if self.modify_position(position, sl=new_sl):
                    logging.info(f"Trailed SL for short position {ticket} to {new_sl}")


class FixedPercentSlTpWithSingleBreakEvenPercentBasedPositionManager(BasePercentBasedPositionManager):
    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        if self.break_even_percent is None:
            raise ValueError(f"{type(self).__name__} requires 'break_even' in position_manager config")
        if self.trailing_percent is None:
            raise ValueError(f"{type(self).__name__} requires 'trailing' in position_manager config")

    def manage_long_position(self, position, flag: bool):
        if flag:
            self._manage_position(position, is_long=True)

    def manage_short_position(self, position, flag: bool):
        if flag:
            self._manage_position(position, is_long=False)

    def _manage_position(self, position, is_long: bool):
        ticket = position["ticket"]
        price_open = position["price_open"]
        price_current = position["price_current"]

        # Set initial SL/TP if they don't exist; use returned effective values so break-even checks in this same cycle
        # see the actual broker state, not the stale sl=0 snapshot from the position dict.
        sl, tp = self._set_initial_sl_tp_percent(position, is_long)

        break_even_threshold = price_open * (self.break_even_percent / 100)
        if is_long:
            break_even_price = price_open + break_even_threshold
            if price_current >= break_even_price and sl < price_open:
                trailing_amount = price_current * (self.trailing_percent / 100)
                new_sl = price_current - trailing_amount
                new_sl = max(new_sl, price_open)
                if new_sl > sl:
                    if self.modify_position(position, sl=new_sl):
                        logging.info(f"Moved SL to break-even/trailing for long position {ticket}")
        else:
            break_even_price = price_open - break_even_threshold
            if price_current <= break_even_price and sl > price_open:
                trailing_amount = price_current * (self.trailing_percent / 100)
                new_sl = price_current + trailing_amount
                new_sl = min(new_sl, price_open)
                if new_sl < sl:
                    if self.modify_position(position, sl=new_sl):
                        logging.info(f"Moved SL to break-even/trailing for short position {ticket}")


class DynamicPercentBasedPositionManager(BasePercentBasedPositionManager):
    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        if self.trailing_percent is None:
            raise ValueError(f"{type(self).__name__} requires 'trailing' in position_manager config")

    def manage_long_position(self, position, flag: bool):
        if flag:
            self._manage_position(position, is_long=True)

    def manage_short_position(self, position, flag: bool):
        if flag:
            self._manage_position(position, is_long=False)

    def _manage_position(self, position, is_long: bool):
        ticket = position["ticket"]
        price_open = position["price_open"]
        price_current = position["price_current"]

        # Set initial SL if not already present
        sl, tp = self._set_initial_sl_tp_percent(position, is_long)

        if is_long and price_current > price_open:
            trailing_amount = price_current * (self.trailing_percent / 100)
            new_sl = price_current - trailing_amount
            if new_sl > sl and new_sl > price_open:
                if self.modify_position(position, sl=new_sl):
                    logging.info(f"Trailed SL for long position {ticket} to {new_sl}")
        elif not is_long and price_current < price_open:
            trailing_amount = price_current * (self.trailing_percent / 100)
            new_sl = price_current + trailing_amount
            if new_sl < sl and new_sl < price_open:
                if self.modify_position(position, sl=new_sl):
                    logging.info(f"Trailed SL for short position {ticket} to {new_sl}")
