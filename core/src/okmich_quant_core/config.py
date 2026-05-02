from enum import Enum
from typing import List, Optional, Dict, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OrderType(Enum):
    NONE = "none"
    BUY = "buy"
    BUY_STOP = "buy_stop"
    BUY_LIMIT = "buy_limit"
    SELL = "sell"
    SELL_STOP = "sell_stop"
    SELL_LIMIT = "sell_limit"


class PositionManagerType(Enum):
    # Point-Based Managers
    FIXED_POINT = "fixed_point"
    FIXED_POINT_WITH_TRAILING = "fixed_point_with_trailing"
    FIXED_POINT_WITH_BREAK_EVEN = "fixed_point_with_break_even"
    DYNAMIC_POINT = "dynamic_point"

    # Percent-Based Managers
    FIXED_PERCENT = "fixed_percent"
    FIXED_PERCENT_WITH_TRAILING = "fixed_percent_with_trailing"
    FIXED_PERCENT_WITH_BREAK_EVEN = "fixed_percent_with_break_even"
    DYNAMIC_PERCENT = "dynamic_percent"

    # ATR-Based Managers
    FIXED_ATR = "fixed_atr"
    FIXED_ATR_WITH_TRAILING = "fixed_atr_with_trailing"
    FIXED_ATR_WITH_BREAK_EVEN = "fixed_atr_with_break_even"
    DYNAMIC_ATR = "dynamic_atr"

    MAX_LOSS_AMOUNT = "max_loss_amount"
    MAX_LOSS_STOP_LOSS = "max_loss_stop_loss"


class PositionManagerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: PositionManagerType = PositionManagerType.FIXED_POINT
    sl: Optional[float] = None
    tp: Optional[float] = None
    point_size: Optional[float] = None
    trailing: Optional[float] = None
    break_even: Optional[float] = None
    atr_period: Optional[int] = None
    max_loss_amount: Optional[float] = None

    @model_validator(mode="after")
    def validate_config_for_type(self) -> "PositionManagerConfig":
        """Validate that required fields are present for each manager type"""
        required_fields: Dict[PositionManagerType, list] = {
            PositionManagerType.FIXED_POINT: ["sl", "tp"],
            PositionManagerType.FIXED_POINT_WITH_TRAILING: ["sl", "tp", "trailing"],
            PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN: ["sl", "tp", "break_even", "trailing",],
            PositionManagerType.DYNAMIC_POINT: ["sl", "trailing"],
            PositionManagerType.FIXED_PERCENT: ["sl", "tp"],
            PositionManagerType.FIXED_PERCENT_WITH_TRAILING: ["sl", "tp", "trailing"],
            PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN: ["sl", "tp", "break_even", "trailing",],
            PositionManagerType.DYNAMIC_PERCENT: ["sl", "trailing"],
            PositionManagerType.FIXED_ATR: ["sl", "tp", "atr_period"],
            PositionManagerType.FIXED_ATR_WITH_TRAILING: ["sl", "tp", "trailing", "atr_period",],
            PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN: ["sl", "tp", "break_even", "trailing", "atr_period",],
            PositionManagerType.DYNAMIC_ATR: ["sl", "trailing", "atr_period"],
            PositionManagerType.MAX_LOSS_AMOUNT: ["max_loss_amount",],
            PositionManagerType.MAX_LOSS_STOP_LOSS: ["max_loss_amount",],
        }

        if self.type in required_fields:
            missing_fields = []
            for field in required_fields[self.type]:
                if getattr(self, field) is None:
                    missing_fields.append(field)

            if missing_fields:
                raise ValueError(f"For {self.type.value}, missing required fields: {missing_fields}")

        return self

    @field_validator("sl", "tp", "trailing", "break_even")
    def validate_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Point/Percent/Atr multiplier values must be positive")
        return v

    @field_validator("atr_period")
    def validate_atr_period(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("ATR period must be positive")
        return v

    @field_validator("max_loss_amount")
    def validate_max_loss_amount(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("Max loss amount must be positive")
        return v


class PositionSizingType(Enum):
    FIXED = "fixed"
    RISK_PCT_OF_EQUITY = "risk_pct_of_equity"
    KELLY_CRITERION = "kelly_criterion"


class PositionSizingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: PositionSizingType = PositionSizingType.FIXED
    units: Optional[float] = 0.01
    risk_pct: Optional[float] = None
    kelly_fraction: Optional[float] = None

    @model_validator(mode="after")
    def validate_config_for_type(self) -> "PositionSizingConfig":
        required_fields: Dict[PositionSizingType, list] = {
            PositionSizingType.FIXED: ["units"],
            PositionSizingType.RISK_PCT_OF_EQUITY: ["risk_pct"],
            PositionSizingType.KELLY_CRITERION: ["kelly_fraction"],
        }
        if self.type in required_fields:
            missing_fields = [f for f in required_fields[self.type] if getattr(self, f) is None]
            if missing_fields:
                raise ValueError(f"For {self.type.value}, missing required fields: {missing_fields}")
        return self

    @field_validator("units")
    def validate_units(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("units must be > 0")
        return v

    @field_validator("risk_pct", "kelly_fraction")
    def validate_fraction(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0 < v <= 1):
            raise ValueError("risk_pct / kelly_fraction must be in (0, 1]")
        return v


class RunLoopConfig(BaseModel):
    sleep_interval: float = 0.5
    chk_position_interval: float = 30


class FilterConfig(BaseModel):
    type: str
    name: Optional[str] = None
    params: dict = Field(default_factory=dict)


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    symbol: str
    timeframe: Union[int, str]
    magic: int
    signal_params: dict = Field(default_factory=dict)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    max_number_of_open_positions: int = 1
    bars_to_copy: int = 100
    position_manager: Optional[PositionManagerConfig] = None
    filters: List[FilterConfig] = Field(default_factory=list)

    def __str__(self):
        return f"StrategyConfig(name={self.name}, symbol={self.symbol}, timeframe={self.timeframe}, magic={self.magic})"


class SystemConfig(BaseModel):
    name: str
    runloop: RunLoopConfig
    strategy: Optional[StrategyConfig] = None
    strategies: List[StrategyConfig] = Field(default_factory=list)

    @field_validator("name")
    def name_must_be_valid(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("SystemConfig.name cannot be empty")
        cleaned_name = v.strip()
        return cleaned_name

    @model_validator(mode="after")
    def ensure_strategy_configuration(self) -> "SystemConfig":
        if self.strategy is None and len(self.strategies) == 0:
            raise ValueError('Must provide either "strategy" or "strategies" (at least one)')

        if self.strategy is not None and len(self.strategies) > 0:
            raise ValueError('Cannot provide both "strategy" and "strategies" - choose one approach')
        return self

    @classmethod
    def load_from_file(cls, file_path) -> "SystemConfig":
        with open(file_path, "r") as file:
            json_data = file.read()

        return SystemConfig.model_validate_json(json_data)
