"""Tayal (2009): a price-volume HHMM over zigzag legs that reads bullish and bearish high-frequency regimes.

Source: A. Tayal, "Regime Switching and Technical Trading with Dynamic Bayesian Networks in High-Frequency Stock
Markets", MMath thesis, University of Waterloo, 2009 (UWSpace 10012/4463), sections 4.3-4.5 and 5.2.

Model (Fig. 4.6). Two top-level regimes, each a two-leaf chain that alternates down and up legs:

- ``R_down`` is entered on a down leg and can only end after a down leg (D U D ... D);
- ``R_up`` is entered on an up leg and can only end after an up leg (U D U ... U);
- the top level always switches when a regime ends (no re-entry), so the leg stream keeps alternating.

Only two transition parameters are free (each regime's exit probability after its entry leg), plus the first leg's
regime. Leaves are pinned to their leg direction through the symbol support.

Observations (Table 4.1). One symbol per leg from direction x trend x flow:

- direction (f0): down / up;
- trend (f1): market structure, higher highs and higher lows over the last five turning points (``STRUCTURE``) or,
  as in the legacy SF build, the leg's size in sigma terciles (``LEG_SIZE``);
- flow (f2): Tayal's volume strengthening / weakening (``VOLUME_RATE``), none (``NONE``, 6 symbols: the structure-only
  baseline and the default), BOCPD break intensity or signed volume terciles.

Tayal's features ignore the size of each move ("price change agnostic", p. 63).

Labelling (p. 87). After fitting, the regime with the higher mean in-sample *block return* (first to last price of
each contiguous run of the regime) is the Run (bullish); the other is the Reversal (bearish). Because ``R_up`` blocks
start and end on up legs, that block return leans positive by construction; ``block_returns_`` and
``leg_drift_`` (mean signed leg move per regime, boundary legs included) are kept so the label can be checked.

Decoding (pp. 67-68). Tayal's live decoder is online Viterbi: at each leg, the end state of the most likely path given
the legs so far (``InferenceMode.CAUSAL_VITERBI``). ``FILTERING`` gives the causal regime probability.

Adaptation to bar data. Tayal's legs are tick-level extrema, i.e. the bid-ask bounce. Here legs are directional-change
swings of a configurable size (``LegThreshold``), so the model reads swings at the horizon you choose.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from ...util import InferenceMode
from ..engine import CategoricalEmission, HHMM, StateNode
from ..inputs.leg_features import (BOCPDBreakScore, DiscreteLegFeature, LegDirection, LegSizeScore,
                                   QuantileLegFeature, SignedVolumeScore, SymbolEncoder, TayalTrend, TayalVolume,
                                   VolumeRateUnit, leg_signs)
from ..inputs.zigzag import LegThreshold, MarketData, Zigzag, ZigzagLegBuilder, legs_to_bars

R_DOWN, R_UP = "R_down", "R_up"
LEAVES = (f"{R_DOWN}.down", f"{R_DOWN}.up", f"{R_UP}.up", f"{R_UP}.down")


class TrendInput(StrEnum):
    """Second symbol component of the Tayal flavour."""

    STRUCTURE = "structure"   # Tayal f1: higher highs / higher lows over five turning points
    LEG_SIZE = "leg_size"     # legacy SF: |leg| / sigma terciles


class FlowInput(StrEnum):
    """Third symbol component of the Tayal flavour."""

    NONE = "none"                    # direction x trend only (6 symbols): the structure-only baseline
    VOLUME_RATE = "volume_rate"      # Tayal f2: volume per second strengthening / weakening (needs volume)
    BOCPD = "bocpd"                  # BOCPD break-intensity terciles (sign-blind: surprise, not flow)
    SIGNED_VOLUME = "signed_volume"  # beta-CLV signed volume with the leg, terciles (needs high/low/volume)


class RegimeLabel(StrEnum):
    BULL = "bull"   # Tayal's Run
    BEAR = "bear"   # Tayal's Reversal


def tayal_tree() -> StateNode:
    """Tayal's Fig. 4.6: R_down enters and ends on its down leg, R_up on its up leg; the top level always switches."""
    r_down = StateNode(R_DOWN, [StateNode.leaf(LEAVES[0]), StateNode.leaf(LEAVES[1])], entry_mask=[True, False],
                       transition_mask=[[False, True], [True, False]], exit_mask=[True, False])
    r_up = StateNode(R_UP, [StateNode.leaf(LEAVES[2]), StateNode.leaf(LEAVES[3])], entry_mask=[True, False],
                     transition_mask=[[False, True], [True, False]], exit_mask=[True, False])
    return StateNode("root", [r_down, r_up], transition_mask=[[False, True], [True, False]])


def tayal_features(trend: TrendInput = TrendInput.STRUCTURE, flow: FlowInput = FlowInput.NONE, *,
                   alpha: float = 0.25, volume_unit: VolumeRateUnit = VolumeRateUnit.SECOND,
                   vol_window: pd.Timedelta | int = pd.Timedelta("24h")) -> list[DiscreteLegFeature]:
    """The symbol components [direction, trend(, flow)] for the chosen inputs."""
    trend, flow = TrendInput(trend), FlowInput(flow)
    features: list[DiscreteLegFeature] = [LegDirection()]
    features.append(TayalTrend() if trend is TrendInput.STRUCTURE
                    else QuantileLegFeature(LegSizeScore(vol_window), 3, name="leg_size"))
    if flow is FlowInput.VOLUME_RATE:
        features.append(TayalVolume(alpha, volume_unit))
    elif flow is FlowInput.BOCPD:
        features.append(QuantileLegFeature(BOCPDBreakScore(vol_window=vol_window), 3, name="bocpd_break"))
    elif flow is FlowInput.SIGNED_VOLUME:
        features.append(QuantileLegFeature(SignedVolumeScore(), 3, name="signed_volume"))
    return features


@dataclass(frozen=True)
class BarLayout:
    """Columns of a bar matrix ``X`` for the duck-typed (numpy) API: positions of close and the optional series."""

    close: int = 0
    volume: Optional[int] = None
    high: Optional[int] = None
    low: Optional[int] = None


class TayalHHMM:
    """Tayal (2009) HHMM on zigzag legs of a configurable size.

    Two ways in:

    - market data: ``fit(MarketData)``, ``decode``, ``bar_regime``, ``legs``: legs on the real clock;
    - the duck-typed ``BasePomegranateHMM`` surface on a bar matrix ``X`` (numpy, columns per ``bar_layout``), as
      ``HmmModelWrapper`` passes it: ``fit(X)``, ``predict_proba(X)``, ``predict_proba_fixed_lag(X, lag)``,
      ``compute_per_bar_predictive_loglik(X)``, ``save`` / ``load`` ...

    The duck-typed surface reports the four leaves (``n_states`` = 4) per *bar*: a leg's posterior holds from its
    confirmation bar until the next leg confirms, and bars before the first confirmed leg carry the entry
    distribution. ``*_level`` methods give the regimes (level 1). The leg size is a fraction of price, so a transform
    pipeline in front of the model must leave prices unscaled (an identity transform).

    Clock. A numpy ``X`` has no timestamps (``HmmModelWrapper`` passes ``.values``), so its bars get a regular clock of
    ``bar_interval``. That is exact only for a configuration that never reads the clock (``reads_the_clock`` False):
    structure trend, no flow or ``VolumeRateUnit.BAR`` volume, and a bar-count ``vol_window`` for leg-size or BOCPD
    inputs. Legs and the frozen threshold do not depend on the clock. A clock-reading configuration refuses a numpy
    matrix rather than invent timestamps (volume per second across a session gap would change); give it a
    ``MarketData`` or a DataFrame with a DatetimeIndex.

    Transitions, ``forecast``, ``expected_duration``, likelihoods and selection scores are per *leg* (event time).
    ``compute_per_bar_predictive_loglik`` puts each leg's predictive log density on its confirmation bar and 0
    elsewhere, so it sums to the leg log-likelihood. A fixed lag of L bars conditions a bar's leg on every leg
    confirmed by t + L.

    Parameters
    ----------
    threshold : leg size (``LegThreshold``): target legs per observed hour, k x sigma, or fixed.
    trend, flow : symbol components (see the module docstring). Defaults: structure, no flow.
    alpha, volume_unit : Tayal's volume-change threshold (25%) and its denominator (per second, or per bar).
    vol_window : realised-vol window for leg-size and BOCPD features.
    bar_layout, bar_interval : how a numpy bar matrix is read, and the clock its bars are given.
    inference_mode : FILTERING (default) or SMOOTHING for ``predict_proba``; ``predict`` follows it, and decodes
        with online Viterbi under CAUSAL_VITERBI or full Viterbi under VITERBI (as ``BasePomegranateHMM``).
    n_init, max_iter, tol, transition_prior, pseudo_count, random_state : passed to the engine and its categorical
        emission.
    """

    def __init__(self, threshold: LegThreshold = LegThreshold.legs_per_hour(1.0),
                 trend: TrendInput = TrendInput.STRUCTURE, flow: FlowInput = FlowInput.NONE, *, alpha: float = 0.25,
                 volume_unit: VolumeRateUnit = VolumeRateUnit.SECOND,
                 vol_window: pd.Timedelta | int = pd.Timedelta("24h"), bar_layout: BarLayout = BarLayout(),
                 bar_interval: pd.Timedelta = pd.Timedelta("5min"),
                 inference_mode: InferenceMode = InferenceMode.FILTERING, n_init: int = 5, max_iter: int = 300,
                 tol: float = 1e-6, transition_prior: float = 1e-3, pseudo_count: float = 1e-3, random_state: int = 0):
        self.threshold = threshold
        self.trend, self.flow = TrendInput(trend), FlowInput(flow)
        self.alpha, self.volume_unit, self.vol_window = float(alpha), VolumeRateUnit(volume_unit), vol_window
        if not isinstance(bar_layout, BarLayout):
            raise TypeError("bar_layout must be a BarLayout")
        self.bar_layout, self.bar_interval = bar_layout, pd.Timedelta(bar_interval)
        if self.bar_interval <= pd.Timedelta(0):
            raise ValueError("bar_interval must be positive")
        self.inference_mode = InferenceMode(inference_mode)
        self.n_init, self.max_iter, self.tol = n_init, max_iter, tol
        self.transition_prior, self.pseudo_count, self.random_state = transition_prior, pseudo_count, random_state
        self.builder_: Optional[ZigzagLegBuilder] = None
        self.encoder_: Optional[SymbolEncoder] = None
        self.model_: Optional[HHMM] = None
        self.labels_: Optional[dict[str, RegimeLabel]] = None
        self.block_returns_: Optional[dict[str, float]] = None
        self.leg_drift_: Optional[dict[str, float]] = None

    _REQUIRED = {FlowInput.VOLUME_RATE: ("volume",), FlowInput.SIGNED_VOLUME: ("high", "low", "volume")}

    #: attributes added after a model may have been saved, with the value an older model behaves as
    _STATE_DEFAULTS = {"bar_layout": BarLayout(), "bar_interval": pd.Timedelta("5min"),
                       "inference_mode": InferenceMode.FILTERING}

    def __setstate__(self, state: dict) -> None:
        for key, value in self._STATE_DEFAULTS.items():
            state.setdefault(key, value)
        self.__dict__.update(state)

    @property
    def reads_the_clock(self) -> bool:
        """True if a symbol depends on timestamps (a time-based vol window, or volume per second)."""
        time_window = isinstance(self.vol_window, pd.Timedelta)
        uses_vol = self.trend is TrendInput.LEG_SIZE or self.flow is FlowInput.BOCPD
        per_second = self.flow is FlowInput.VOLUME_RATE and self.volume_unit is VolumeRateUnit.SECOND
        return (uses_vol and time_window) or per_second

    # ==================================================================================================================
    # Fitting
    # ==================================================================================================================
    def fit(self, data: "MarketData | np.ndarray | pd.DataFrame",
            lengths: Optional[Sequence[int]] = None) -> "TayalHHMM":
        """Freeze the leg size and feature boundaries on the training bars, fit the HHMM, label bull / bear.

        ``data`` is a ``MarketData``, or a bar matrix read with ``bar_layout``. Everything is built locally and
        published together only when the fit succeeds, so a failed refit leaves the previous fit intact.
        """
        if lengths is not None:
            raise NotImplementedError("TayalHHMM fits one contiguous bar series; lengths are not supported")
        market_data = self._market(data)
        if self.flow in self._REQUIRED:
            market_data.require(f"FlowInput.{self.flow.name}", *self._REQUIRED[self.flow])
        builder = ZigzagLegBuilder(self.threshold).fit(market_data.close)
        legs = builder.transform(market_data.close)
        if len(legs) < 10:
            raise ValueError(f"Only {len(legs)} legs on the training bars; use a smaller leg size or more data")
        features = tayal_features(self.trend, self.flow, alpha=self.alpha, volume_unit=self.volume_unit,
                                  vol_window=self.vol_window)
        encoder = SymbolEncoder(features).fit(legs, market_data)
        symbols = encoder.transform(legs, market_data)
        down, up = encoder.symbols_with("direction", 0), encoder.symbols_with("direction", 1)
        support = {LEAVES[0]: down, LEAVES[1]: up, LEAVES[2]: up, LEAVES[3]: down}
        emission = CategoricalEmission(0, encoder.n_symbols, support=support, pseudo_count=self.pseudo_count)
        model = HHMM(tayal_tree(), [emission], n_init=self.n_init, max_iter=self.max_iter, tol=self.tol,
                     transition_prior=self.transition_prior, random_state=self.random_state)
        X = symbols[:, None].astype(np.float64)
        model.fit(X)
        labels, block_returns, leg_drift = _label(model, legs, model.leaf_to_level(model.predict_viterbi(X), 1))
        self.builder_, self.encoder_, self.model_ = builder, encoder, model
        self.labels_, self.block_returns_, self.leg_drift_ = labels, block_returns, leg_drift
        return self

    def fit_predict(self, data, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        return self.fit(data, lengths).predict(data)

    def train(self, data, lengths: Optional[Sequence[int]] = None, **kwargs) -> "TayalHHMM":
        return self.fit(data, lengths)

    # ==================================================================================================================
    # Market-data API
    # ==================================================================================================================
    def legs(self, market_data: "MarketData | np.ndarray | pd.DataFrame") -> list[Zigzag]:
        """Legs of the bars at the frozen leg size."""
        self._check_fitted()
        return self.builder_.transform(self._market(market_data).close)

    def decode(self, market_data: "MarketData | np.ndarray | pd.DataFrame",
               mode: InferenceMode = InferenceMode.CAUSAL_VITERBI,
               start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Per-leg regime read, one row per leg confirmed at or after ``start`` (default: all), indexed by
        confirmation time.

        The decoder runs over every leg of ``market_data`` from its beginning, so supply history before ``start``. With
        a causal ``mode`` (CAUSAL_VITERBI, FILTERING) each row uses only the legs up to it. ``p_bull`` is always the
        filtered (causal) probability of the bull regime.
        """
        self._check_fitted()
        mode = InferenceMode(mode)
        md = self._market(market_data)
        legs = self.builder_.transform(md.close)
        if not legs:                                  # e.g. a live feed before its first confirmed swing
            return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _DECODE_COLUMNS.items()},
                                index=pd.DatetimeIndex([], name="confirm_time", tz=md.close.index.tz))
        X = self._symbols(legs, md)
        filt = self.model_.filter(X)
        names = self.model_.level_names(1)
        p_level = self.model_.level_marginal(filt, 1)
        leaf = self._decode_leaves(X, mode, filt)
        regime = self.model_.leaf_to_level(leaf, 1)
        bull = names.index(self._bull_name())
        frame = pd.DataFrame({
            "confirm_bar": [z.confirm_bar for z in legs],
            "direction": leg_signs(legs),
            "symbol": X[:, 0].astype(np.int64),
            "leaf": [self.model_.topology.leaf_names[j] for j in leaf],
            "regime": [names[r] for r in regime],
            "label": [self.labels_[names[r]].value for r in regime],
            "p_bull": p_level[:, bull],
        }, index=pd.DatetimeIndex([z.end_time for z in legs], name="confirm_time"))
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        return frame

    def bar_regime(self, market_data: "MarketData | np.ndarray | pd.DataFrame",
                   mode: InferenceMode = InferenceMode.CAUSAL_VITERBI, delay: int = 1) -> pd.Series:
        """+1 (bull) / -1 (bear) on every bar, held from each leg's confirmation bar + ``delay`` (NaN before the first
        confirmed leg, so all NaN when there is none yet)."""
        md = self._market(market_data)
        frame = self.decode(md, mode)
        legs = self.builder_.transform(md.close)
        if not legs:
            return pd.Series(np.nan, index=md.close.index, name="tayal_regime")
        values = np.where(frame["label"].to_numpy() == RegimeLabel.BULL.value, 1.0, -1.0)
        bars = legs_to_bars(values, legs, len(md.close), delay=delay)
        return pd.Series(bars, index=md.close.index, name="tayal_regime")

    # ==================================================================================================================
    # Duck-typed BasePomegranateHMM surface (per bar; states = the four leaves)
    # ==================================================================================================================
    @property
    def n_states(self) -> int:
        return len(LEAVES)

    def predict_proba(self, X) -> np.ndarray:
        """(T, 4) leaf posteriors per bar under ``inference_mode`` (FILTERING or SMOOTHING)."""
        return self.state_posteriors(X)

    def state_posteriors(self, X) -> np.ndarray:
        md, legs, S = self._prepare(X)
        if self.inference_mode is InferenceMode.FILTERING:
            per_leg = self.model_.filter(S) if legs else None
        elif self.inference_mode is InferenceMode.SMOOTHING:
            per_leg = self.model_.smooth(S) if legs else None
        else:
            raise ValueError(f"{self.inference_mode} gives a path, not posteriors; use predict() or "
                             f"FILTERING/SMOOTHING")
        return self._hold(per_leg, legs, len(md.close), self.model_.initial_distribution())

    def predict(self, X) -> np.ndarray:
        """(T,) leaf labels per bar: argmax of the posteriors, or the decoded leg path under the Viterbi modes."""
        if self.inference_mode is InferenceMode.CAUSAL_VITERBI:
            return self.predict_causal_viterbi(X)
        if self.inference_mode is InferenceMode.VITERBI:
            return self.predict_viterbi(X)
        return self.predict_proba(X).argmax(axis=1)

    def predict_viterbi(self, X) -> np.ndarray:
        """(T,) leaves of the most likely leg path given every leg: hindsight only."""
        return self._hold_labels(X, self.model_.predict_viterbi)

    def predict_causal_viterbi(self, X) -> np.ndarray:
        """(T,) leaf ending the most likely leg path given the legs confirmed so far (Tayal's online decoder)."""
        return self._hold_labels(X, self.model_.predict_causal_viterbi)

    def causal_viterbi_scores(self, X) -> np.ndarray:
        """(T, 4) max-product scores of the latest confirmed leg (the log entry distribution before the first)."""
        md, legs, S = self._prepare(X)
        per_leg = self.model_.causal_viterbi_scores(S) if legs else None
        with np.errstate(divide="ignore"):
            prefix = np.log(self.model_.initial_distribution())
        return self._hold(per_leg, legs, len(md.close), prefix)

    def predict_proba_fixed_lag(self, X, lag: int) -> np.ndarray:
        """(T, 4) posterior of each bar's leg given every leg confirmed by bar t + lag (lag in bars, open-ended)."""
        return self.predict_proba_fixed_lag_sweep(X, [lag])[_check_lag(lag)]

    def predict_fixed_lag(self, X, lag: int) -> np.ndarray:
        return self.predict_proba_fixed_lag(X, lag).argmax(axis=1)

    def predict_proba_fixed_lag_sweep(self, X, lags: Sequence[int],
                                      window_size: Optional[int] = None) -> dict[int, np.ndarray]:
        """{lag: (T, 4)} for several bar lags. ``window_size`` (a bounded bar buffer) is not supported: legs depend on
        the whole bar history through the leg detector."""
        if window_size is not None:
            raise NotImplementedError("TayalHHMM does not support bounded-history (window_size) fixed lag")
        unique = sorted({_check_lag(lag) for lag in lags})
        md, legs, S = self._prepare(X)
        n_bars = len(md.close)
        prefix = self.model_.initial_distribution()
        if not legs:
            return {lag: np.tile(prefix, (n_bars, 1)) for lag in unique}
        confirm = np.array([z.confirm_bar for z in legs])
        current = np.searchsorted(confirm, np.arange(n_bars), side="right") - 1      # leg active at each bar
        out: dict[int, np.ndarray] = {}
        cache: dict[int, np.ndarray] = {}
        for lag in unique:
            ahead = np.searchsorted(confirm, np.minimum(np.arange(n_bars) + lag, n_bars - 1), side="right") - 1
            res = np.tile(prefix, (n_bars, 1))
            for m in np.unique((ahead - current)[current >= 0]):
                if m not in cache:
                    cache[m] = self.model_.predict_proba_fixed_lag(S, int(m))
                rows = (current >= 0) & (ahead - current == m)
                res[rows] = cache[m][current[rows]]
            out[lag] = res
        return out

    def fixed_lag_diagnostics(self, X, lag: int) -> dict:
        posteriors = self.predict_proba_fixed_lag(X, lag)
        log_p = np.log(np.maximum(posteriors, 1e-300))
        return {"posteriors": posteriors, "map_labels": posteriors.argmax(axis=1),
                "max_posterior": posteriors.max(axis=1), "entropy": -np.sum(posteriors * log_p, axis=1),
                "posterior_delta": np.diff(posteriors, axis=0, prepend=posteriors[:1])}

    def log_likelihood(self, X) -> float:
        """Log-likelihood of the bars' legs (0 when there is no confirmed leg)."""
        _, legs, S = self._prepare(X)
        return self.model_.log_likelihood(S) if legs else 0.0

    def compute_per_bar_predictive_loglik(self, X) -> np.ndarray:
        """(T,) each leg's one-step predictive log density on its confirmation bar, 0 on other bars."""
        md, legs, S = self._prepare(X)
        out = np.zeros(len(md.close))
        if legs:
            out[[z.confirm_bar for z in legs]] = self.model_.compute_per_bar_predictive_loglik(S)
        return out

    def path_entropy(self, X) -> float:
        _, legs, S = self._prepare(X)
        return self.model_.path_entropy(S) if legs else 0.0

    def selection_scores(self, X, lengths: Optional[Sequence[int]] = None, *, with_entropy: bool = True):
        """AIC / BIC / ICL over the bars' legs (n_obs = number of legs)."""
        if lengths is not None:
            raise NotImplementedError("TayalHHMM scores one contiguous bar series; lengths are not supported")
        _, legs, S = self._prepare(X)
        if not legs:
            raise ValueError("No confirmed leg in X: nothing to score")
        return self.model_.selection_scores(S, with_entropy=with_entropy)

    def get_aic_bic(self, X) -> tuple[float, float]:
        scores = self.selection_scores(X, with_entropy=False)
        return scores.aic, scores.bic

    def transition_prob(self) -> np.ndarray:
        """(4, 4) leaf transition matrix per leg."""
        self._check_fitted()
        return self.model_.transition_prob()

    def forecast(self, posterior: Optional[np.ndarray] = None, n_steps: int = 1,
                 current_regime: Optional[int] = None) -> np.ndarray:
        """Leaf probabilities ``n_steps`` legs ahead."""
        self._check_fitted()
        return self.model_.forecast(posterior, n_steps, current_regime)

    def get_stationary_distribution(self) -> np.ndarray:
        self._check_fitted()
        return self.model_.get_stationary_distribution()

    def expected_duration(self, regime: int) -> float:
        """Expected legs per visit to a leaf (1 for Tayal's alternating legs)."""
        self._check_fitted()
        return self.model_.expected_duration(regime)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TayalHHMM":
        return joblib.load(path)

    # ==================================================================================================================
    # Levels (level 1 = the regimes R_down / R_up)
    # ==================================================================================================================
    def level_names(self, level: int = 1) -> list[str]:
        self._check_fitted()
        return self.model_.level_names(level)

    def predict_proba_level(self, X, level: int = 1) -> np.ndarray:
        """(T, n_level_nodes) per-bar posterior of the nodes at ``level``."""
        self._check_fitted()
        return self.model_.level_marginal(self.predict_proba(X), level)

    def predict_level(self, X, level: int = 1) -> np.ndarray:
        self._check_fitted()
        if self.inference_mode in (InferenceMode.CAUSAL_VITERBI, InferenceMode.VITERBI):
            return self.model_.leaf_to_level(self.predict(X), level)
        return self.predict_proba_level(X, level).argmax(axis=1)

    def predict_proba_fixed_lag_level(self, X, lag: int, level: int = 1) -> np.ndarray:
        self._check_fitted()
        return self.model_.level_marginal(self.predict_proba_fixed_lag(X, lag), level)

    # ==================================================================================================================
    # Internals
    # ==================================================================================================================
    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("TayalHHMM is not fitted; call fit() first")

    def _bull_name(self) -> str:
        return next(n for n, lab in self.labels_.items() if lab is RegimeLabel.BULL)

    def _market(self, data) -> MarketData:
        """A ``MarketData`` as given; a DataFrame keeps its DatetimeIndex; a numpy matrix gets a regular clock."""
        if isinstance(data, MarketData):
            return data
        layout = self.bar_layout
        if isinstance(data, pd.DataFrame):
            values = data.to_numpy(dtype=np.float64)
            index = data.index if isinstance(data.index, pd.DatetimeIndex) else None
        else:
            values = np.asarray(data, dtype=np.float64)
            index = None
        if values.ndim != 2:
            raise ValueError(f"A bar matrix must be 2-D (bars x columns), got shape {values.shape}")
        used = [c for c in (layout.close, layout.volume, layout.high, layout.low) if c is not None]
        if values.shape[1] <= max(used):
            raise ValueError(f"The bar matrix has {values.shape[1]} columns; bar_layout reads column {max(used)}")
        if index is None:
            if self.reads_the_clock:
                raise ValueError("This TayalHHMM reads the clock (a time-based vol_window for leg-size / BOCPD inputs, "
                                 "or volume per second), but a numpy bar matrix has no timestamps. Pass a MarketData "
                                 "or a DataFrame with a DatetimeIndex, or configure it clock-free (vol_window as a bar "
                                 "count, volume_unit=VolumeRateUnit.BAR) for pipelines that pass numpy.")
            index = pd.date_range("2000-01-03", periods=len(values), freq=self.bar_interval, tz="UTC")

        def column(c):
            return None if c is None else pd.Series(values[:, c], index=index)

        return MarketData(close=column(layout.close), volume=column(layout.volume), high=column(layout.high),
                          low=column(layout.low))

    def _symbols(self, legs: list[Zigzag], md: MarketData) -> np.ndarray:
        return self.encoder_.transform(legs, md)[:, None].astype(np.float64)

    def _prepare(self, X) -> tuple[MarketData, list[Zigzag], Optional[np.ndarray]]:
        self._check_fitted()
        md = self._market(X)
        legs = self.builder_.transform(md.close)
        return md, legs, (self._symbols(legs, md) if legs else None)

    @staticmethod
    def _hold(per_leg: Optional[np.ndarray], legs: list[Zigzag], n_bars: int, prefix: np.ndarray) -> np.ndarray:
        """Per-leg rows held on bars from each confirmation; ``prefix`` before the first confirmed leg."""
        out = np.tile(np.asarray(prefix, dtype=np.float64), (n_bars, 1))
        if legs:
            held = legs_to_bars(per_leg, legs, n_bars, delay=0)
            started = np.arange(n_bars) >= legs[0].confirm_bar     # by position: a -inf score is a valid value
            out[started] = held[started]
        return out

    def _hold_labels(self, X, decoder) -> np.ndarray:
        md, legs, S = self._prepare(X)
        n_bars = len(md.close)
        out = np.full(n_bars, int(np.argmax(self.model_.initial_distribution())), dtype=np.int64)
        if legs:
            held = legs_to_bars(decoder(S).astype(np.float64), legs, n_bars, delay=0)
            started = np.arange(n_bars) >= legs[0].confirm_bar
            out[started] = held[started].astype(np.int64)
        return out

    def _decode_leaves(self, X: np.ndarray, mode: InferenceMode, filt: np.ndarray) -> np.ndarray:
        if mode is InferenceMode.CAUSAL_VITERBI:
            return self.model_.predict_causal_viterbi(X)
        if mode is InferenceMode.VITERBI:
            return self.model_.predict_viterbi(X)
        if mode is InferenceMode.FILTERING:
            return filt.argmax(axis=1)
        return self.model_.smooth(X).argmax(axis=1)

    def __repr__(self) -> str:
        return (f"TayalHHMM(threshold={self.threshold.mode.value}={self.threshold.value}, trend={self.trend.value}, "
                f"flow={self.flow.value}, fitted={self.model_ is not None})")


_DECODE_COLUMNS = {"confirm_bar": "int64", "direction": "int64", "symbol": "int64", "leaf": "object",
                   "regime": "object", "label": "object", "p_bull": "float64"}


def _check_lag(lag) -> int:
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError(f"lag must be a non-negative integer, got {lag!r}")
    return int(lag)


def _label(model: HHMM, legs: list[Zigzag],
           regime_path: np.ndarray) -> tuple[dict[str, RegimeLabel], dict[str, float], dict[str, float]]:
    """Tayal's rule: the regime with the higher mean block return is bull (in-sample Viterbi path). Also returns the
    block returns and the mean signed leg move per regime, so the label can be checked."""
    names = model.level_names(1)
    starts = np.flatnonzero(np.r_[True, np.diff(regime_path) != 0])
    ends = np.r_[starts[1:] - 1, len(regime_path) - 1]
    returns: dict[str, list[float]] = {n: [] for n in names}
    for s, e in zip(starts, ends):
        returns[names[regime_path[s]]].append(legs[e].end_price / legs[s].start_price - 1.0)
    block_returns = {n: float(np.mean(r)) if r else float("nan") for n, r in returns.items()}
    signed = leg_signs(legs) * np.array([z.magnitude for z in legs])
    leg_drift = {n: float(signed[regime_path == k].mean()) if np.any(regime_path == k) else float("nan")
                 for k, n in enumerate(names)}
    a, b = (block_returns[n] for n in names)
    bull = names[0] if (np.isnan(b) or (not np.isnan(a) and a > b)) else names[1]
    return {n: RegimeLabel.BULL if n == bull else RegimeLabel.BEAR for n in names}, block_returns, leg_drift
