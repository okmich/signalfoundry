# okmich_quant_ml.hmm.hierarchical

Hierarchical hidden Markov models: a general engine, observation builders, and paper presets ("flavours").

```
engine/     HHMM: any state tree, exact inference by flattening, EM on the tree's own parameters
inputs/     zigzag legs of a configurable size, per-leg features, a symbol encoder
flavours/   TayalHHMM (Tayal 2009), joint_multiscale, embedded_hhmm (Adam et al. 2019)
```

## The engine

A tree of `StateNode`s. Leaves emit. Every internal node q has three parameter sets over its children:

| Parameter | Meaning |
|---|---|
| `entry` pi_q(c) | when q is activated, the child it starts in |
| `horizontal` A_q(c, c') | given q does not end after child c, the next child |
| `exit` e_q(c) | the probability that q ends (hands control to its parent) after child c |

The root never ends. Masks (`entry_mask`, `transition_mask`, `exit_mask`) put structural zeros anywhere: a regime that
can only be entered and left through one child, a left-to-right chain, no re-entry of the same top state.

**Inference.** The tree flattens onto its N leaves (Murphy & Paskin 2001): leaf i to leaf j sums, over the ancestors of
i where the horizontal move can happen, exits below x no exit there x A x entries down to j. Inference on the leaf chain
is exact at O(T N^2), in numba, in log space (a state ruled out for hundreds of bars stays recoverable).

**Learning.** EM treats the level at which each transition happened as latent: each pairwise posterior is split over
its paths, giving expected counts of entries, horizontal moves, exits and non-exits, and closed-form M-steps for
pi, A and e (optional Dirichlet/Beta pseudo-counts `transition_prior`). Structural zeros stay zero. This learns the
tree's parameters, not free flat cells. `fit_report.objective` (what EM maximises and restarts are ranked on) includes
the log priors implied by `transition_prior` and the categorical `pseudo_count`; `fit_report.loglik` is the raw
likelihood. `fit` is atomic (a failed fit leaves the previous one in place), and warm starts must match the tree, its
masks and the emission configuration exactly.

**Exit schedules.** `ExitSchedule` sets, per depth and bar, whether nodes may end (`LEARNED`), cannot (`FORBIDDEN`) or
must (`FORCED`). FORCED at each day's first bar and FORBIDDEN inside gives the embedded HHMM: a day state that owns an
intraday chain. Pass a `schedule`, or give the model a `schedule_column` holding a block id (e.g. the trading day) and
it rebuilds the schedule from X itself, so a pipeline that only passes a feature matrix still runs it correctly (an
injective transform such as a scaler keeps the block boundaries). Exits at depths a fit never saw as LEARNED are
excluded from the parameter count, and decoding with them LEARNED warns.

**Emissions.** `CategoricalEmission` (with per-leaf allowed symbols), `GaussianEmission` and `StudentTEmission`
(diagonal or full; Student-t by ECM with learnt degrees of freedom). Blocks multiply; `tie_level=d` shares a block's
parameters among leaves with the same depth-d ancestor, so slow features can inform the top level and fast features
the leaves. NaN means unobserved. Each column belongs to one block, and the model deep-copies the blocks it is given
(read fitted parameters from `model.emissions`).

## Duck typing: the BasePomegranateHMM surface

Every model here (the engine and every flavour) answers `BasePomegranateHMM`'s public methods with the **leaves** as
its states (`n_states` = number of leaves), so it plugs into `HmmModelWrapper`, `compute_trajectories` (adaptive lag)
and the posterior-inference stack:

| Group | Methods |
|---|---|
| fit | `fit`, `fit_predict`, `train` |
| posteriors | `predict_proba`, `state_posteriors` (FILTERING / SMOOTHING per `inference_mode`), `predict` |
| fixed lag | `predict_proba_fixed_lag`, `predict_fixed_lag`, `predict_proba_fixed_lag_sweep` (full or bounded `window_size`), `fixed_lag_diagnostics` |
| Viterbi | `predict_viterbi` (hindsight), `predict_causal_viterbi`, `causal_viterbi_scores` |
| likelihood | `log_likelihood`, `compute_per_bar_predictive_loglik`, `path_entropy`, `selection_scores`, `get_aic_bic` |
| chain | `transition_prob`, `forecast`, `get_stationary_distribution`, `expected_duration` |
| files | `save`, `load` (joblib; an attribute added to a class since a model was saved loads at its default) |

The same reads for any tree level (level 1 = the top-level states) are separate functions: `predict_proba_level`,
`predict_level`, `predict_proba_fixed_lag_level`, `stationary_level_distribution`, `expected_level_duration` (bars per
contiguous visit to a node), plus `level_names`, `level_marginal` and `leaf_to_level`. Leaf-level `filter` and
`smooth` remain. `tests/hmm/hierarchical/test_duck_typing.py` checks every method against `PomegranateHMM` and runs
each model through `HmmModelWrapper` end to end.

```python
from okmich_quant_ml.hmm.hierarchical import HHMM, StateNode, GaussianEmission

regime = lambda n: StateNode(n, [StateNode.leaf(f"{n}.a"), StateNode.leaf(f"{n}.b")])
root = StateNode("root", [regime("bull"), regime("bear")], transition_mask=[[False, True], [True, False]])
model = HHMM(root, [GaussianEmission([0], tie_level=1), GaussianEmission([1])]).fit(X)   # col 0 slow, col 1 fast
p_leaves = model.predict_proba(X)                 # (T, 4): BasePomegranateHMM-compatible
p_regime = model.predict_proba_level(X, level=1)  # (T, 2): causal P(bull), P(bear) per bar
```

Validated against independent oracles (tests in `tests/hmm/hierarchical/engine`):
- the flat matrix equals the recursive HHMM semantics under every exit mode;
- likelihood, filter, smoother and both Viterbis equal brute-force path enumeration (path entropy and Viterbi scores
  too);
- a depth-1 tree is a plain HMM, and the bounded-window fixed lag equals the base class's algorithm;
- the embedded schedule equals Adam et al.'s block likelihood (eq. 5);
- EM is monotone and reaches the closed-form count optimum when the leaf path is observed;
- every causal output is prefix-invariant.

## Inputs (event time)

`ZigzagLegBuilder(LegThreshold)` cuts prices into legs. The leg size is always a parameter:
`LegThreshold.legs_per_hour(1.0)`, `.k_sigma(2.0)` or `.fixed(0.001)`. Rates are per *observed* hour (bars x bar
interval), so weekends and removed sessions do not dilute them. The threshold is frozen at `fit` and reused by
`transform`, and legs confirmed by bar t never change when later bars arrive. `calibrate_k` warns when the target rate
is unreachable (e.g. several legs per hour on 5m bars is bar noise). `MarketData` requires every series to share
`close`'s exact index, and the features check `0 <= start <= end <= confirm < n` and that each leg's bar positions
match its timestamps before reading.

Per-leg features: `LegDirection`, `TayalTrend` (Tayal f1: higher highs and higher lows over five turning points),
`TayalVolume` (f2: volume strengthening / weakening, alpha = 25%), and `QuantileLegFeature` over `LegSizeScore`,
`BOCPDBreakScore` (changepoint intensity; sign-blind) or `SignedVolumeScore`. `SymbolEncoder` combines them;
`leg_feature_frame` gives continuous per-leg features; `legs_to_bars` maps per-leg values onto bars from each leg's
confirmation.

## Flavours

| Flavour | Tree | Inputs | Read |
|---|---|---|---|
| `TayalHHMM` | `R_down` (D...D) and `R_up` (U...U), top level always switches; 2 free transition parameters | direction x structure (6 symbols, default) or x flow (18) | online Viterbi; bull = higher mean in-sample block return |
| `joint_multiscale` | backdrop states x fast leaves, no re-entry | slow columns tied at depth 1, fast per leaf | `predict_proba_level(X, 1)` |
| `embedded_hhmm` (+ `block_column` or `block_schedule`) | day states owning intraday chains | bar features (+ optional day-level columns) | P(day state) through the day |

**`TayalHHMM` is event-time.** Its market-data API (`fit(MarketData)`, `decode`, `bar_regime`) works on legs with the
real clock. Its duck-typed API works on a bar matrix read with `bar_layout` (close, optional volume / high / low): a
leg's posterior holds on bars from its confirmation until the next leg, bars before the first leg carry the entry
distribution, and a fixed lag of L bars conditions on every leg confirmed by t + L. Transitions, forecasts,
likelihoods and selection scores are per leg; `compute_per_bar_predictive_loglik` puts each leg's predictive log
density on its confirmation bar (0 elsewhere). Prices must reach it unscaled (an identity transform pipeline), and the
bounded-window lag sweep is not supported (pass `window_size=None` to `compute_trajectories`).

`HmmModelWrapper` passes `.values`, so a numpy matrix has no timestamps and its bars get a regular `bar_interval`
clock. That is exact only when the configuration never reads the clock (`reads_the_clock` is False): structure trend,
no flow or `VolumeRateUnit.BAR` volume, and a bar-count `vol_window` for leg-size or BOCPD inputs (legs and the frozen
threshold never depend on the clock). A clock-reading configuration refuses a numpy matrix rather than invent
timestamps; give it `MarketData` or a DataFrame with a DatetimeIndex.
