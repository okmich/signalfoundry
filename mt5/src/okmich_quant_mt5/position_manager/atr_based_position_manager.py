import logging

from .base import BaseMt5PositionManager
from ..functions import get_atr
from okmich_quant_core import StrategyConfig


class BaseAtrBasedPositionManager(BaseMt5PositionManager):
    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        config = strategy_config.position_manager
        self.atr_multiplier_sl = config.sl if config.sl else 0.0
        self.atr_multiplier_tp = config.tp if config.tp else 0.0
        self.atr_multiplier_trailing = config.trailing
        self.atr_multiplier_break_even = config.break_even
        self.atr_period = config.atr_period
        self.atr_timeframe = strategy_config.timeframe

    def _get_current_atr(self):
        return get_atr(self.symbol, self.atr_timeframe, self.atr_period)

    def _set_initial_sl_tp_atr(self, position, is_long: bool, atr: float = None) -> tuple[float, float]:
        """
        Set initial SL/TP on the broker if not already present.

        If atr is provided it is used directly, avoiding a redundant API call when
        the caller has already fetched ATR for the same cycle.

        Returns (effective_sl, effective_tp) — the values that are now (or were already) on the broker, so callers can
        use them immediately without waiting for a position refresh next cycle.  On ATR failure or modification failure
        the original (stale) values are returned.
        """
        ticket = position["ticket"]
        sl = position["sl"]
        tp = position["tp"]
        price_open = position["price_open"]

        # Skip if nothing left to set:
        # - both are present, or
        # - sl is present and no TP is configured (tp will never be set)
        if sl != 0 and (tp != 0 or self.atr_multiplier_tp == 0):
            return sl, tp

        if atr is None:
            atr = self._get_current_atr()
        if atr is None or atr <= 0:
            logging.error(f"Could not get ATR for {self.symbol}")
            return sl, tp  # ATR failed; return original stale values

        new_sl = 0.0
        new_tp = 0.0

        if sl == 0 and self.atr_multiplier_sl > 0:
            if is_long:
                new_sl = price_open - (atr * self.atr_multiplier_sl)
            else:
                new_sl = price_open + (atr * self.atr_multiplier_sl)

        if tp == 0 and self.atr_multiplier_tp > 0:
            if is_long:
                new_tp = price_open + (atr * self.atr_multiplier_tp)
            else:
                new_tp = price_open - (atr * self.atr_multiplier_tp)

        if new_sl <= 0 and sl == 0 and self.atr_multiplier_sl > 0:
            logging.error(f"Computed SL <= 0 for {self.symbol} position {ticket}: {new_sl}. SL will not be set.")
            new_sl = 0.0
        if new_tp <= 0 and tp == 0 and self.atr_multiplier_tp > 0:
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


class FixedAtrBasedPositionManager(BaseAtrBasedPositionManager):
    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)

    def manage_long_position(self, position, flag: bool):
        if flag:
            self._manage_position(position, is_long=True)

    def manage_short_position(self, position, flag: bool):
        if flag:
            self._manage_position(position, is_long=False)

    def _manage_position(self, position, is_long: bool):
        self._set_initial_sl_tp_atr(position, is_long)


class FixedAtrSlTpWithTrailingAtrBasedPositionManager(BaseAtrBasedPositionManager):
    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        if self.atr_multiplier_trailing is None:
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

        atr = self._get_current_atr()
        if atr is None or atr <= 0:
            logging.error(f"Could not get ATR for {self.symbol}")
            return

        # Pass atr so _set_initial_sl_tp_atr reuses it instead of fetching again
        sl, tp = self._set_initial_sl_tp_atr(position, is_long, atr=atr)

        if is_long and price_current > price_open:
            trailing_amount = atr * self.atr_multiplier_trailing
            new_sl = price_current - trailing_amount
            if new_sl > sl:
                if self.modify_position(position, sl=new_sl):
                    logging.info(f"Trailed SL for long position {ticket} to {new_sl}")
        elif not is_long and price_current < price_open:
            trailing_amount = atr * self.atr_multiplier_trailing
            new_sl = price_current + trailing_amount
            if new_sl < sl:
                if self.modify_position(position, sl=new_sl):
                    logging.info(f"Trailed SL for short position {ticket} to {new_sl}")


class FixedAtrSlTpWithSingleBreakEvenAtrBasedPositionManager(BaseAtrBasedPositionManager):
    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        if self.atr_multiplier_break_even is None:
            raise ValueError(f"{type(self).__name__} requires 'break_even' in position_manager config")
        if self.atr_multiplier_trailing is None:
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

        atr = self._get_current_atr()
        if atr is None or atr <= 0:
            logging.error(f"Could not get ATR for {self.symbol}")
            return

        # Pass atr so _set_initial_sl_tp_atr reuses it instead of fetching again
        sl, tp = self._set_initial_sl_tp_atr(position, is_long, atr=atr)

        break_even_threshold = atr * self.atr_multiplier_break_even

        if is_long:
            break_even_price = price_open + break_even_threshold
            if price_current >= break_even_price and sl < price_open:
                trailing_amount = atr * self.atr_multiplier_trailing
                new_sl = price_current - trailing_amount
                new_sl = max(new_sl, price_open)
                if new_sl > sl:
                    if self.modify_position(position, sl=new_sl):
                        logging.info(
                            f"Moved SL to break-even/trailing for long position {ticket}"
                        )
        else:
            break_even_price = price_open - break_even_threshold
            if price_current <= break_even_price and sl > price_open:
                trailing_amount = atr * self.atr_multiplier_trailing
                new_sl = price_current + trailing_amount
                new_sl = min(new_sl, price_open)
                if new_sl < sl:
                    if self.modify_position(position, sl=new_sl):
                        logging.info(
                            f"Moved SL to break-even/trailing for short position {ticket}"
                        )


class DynamicAtrBasedPositionManager(BaseAtrBasedPositionManager):
    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        if self.atr_multiplier_trailing is None:
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

        atr = self._get_current_atr()
        if atr is None or atr <= 0:
            logging.error(f"Could not get ATR for {self.symbol}")
            return

        # Set initial SL if not already present; reuse fetched ATR
        sl, tp = self._set_initial_sl_tp_atr(position, is_long, atr=atr)

        if is_long and price_current > price_open:
            trailing_amount = atr * self.atr_multiplier_trailing
            new_sl = price_current - trailing_amount
            if new_sl > sl and new_sl > price_open:
                if self.modify_position(position, sl=new_sl):
                    logging.info(f"Trailed SL for long position {ticket} to {new_sl}")
        elif not is_long and price_current < price_open:
            trailing_amount = atr * self.atr_multiplier_trailing
            new_sl = price_current + trailing_amount
            if new_sl < sl and new_sl < price_open:
                if self.modify_position(position, sl=new_sl):
                    logging.info(f"Trailed SL for short position {ticket} to {new_sl}")
