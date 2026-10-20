# regime_diagnostics

What follows each state of a causal regime label? This package answers that with one call.

`label_path_structure_statistics` describes what happens inside each state, looking back. This package asks what
follows each state, looking forward. It runs both, and reports what to expect next per state, optionally per cell
of another regime dimension.

```python
from okmich_quant_labelling.utils.regime_diagnostics import regime_diagnostics, seasonal_cost

diag = regime_diagnostics(close, label, first_test_start="2024-07-01", condition={"vol": vol_state},
                          name="EURUSD path | vol", cost=seasonal_cost(spread_frac, fit_until="2024-07-01"),
                          label_fit_end=fit_end, out_dir="results/diagnostics/EURUSD_path_given_vol")
print(diag.story)        # plain-language read: numbers with bands, caveats where the null cannot decide
diag.expectations        # per label state x condition cell: mean, q05..q95 with 90% day-block bands
diag.report_path         # one HTML: story, label, descriptive, forward distributions, tails, conditional, verdicts
```

Read the per-state **distributions** first (`diag.forward.quantiles`, the expectation table). The verdicts only
summarise them.

## Inputs

- `close` and `label` share one tz-aware index on the true clock. Convert broker stamps first.
- `label[t]` is the state known at the close of bar t; NaN means unknown.
- `label_fit_end` is a timestamp, or a per-bar Series for a label refitted walk-forward. A label fitted on
  out-of-sample bars is flagged DIAGNOSTIC, and nothing in that run is tradeable.
- `cost` is the round-trip cost per decision bar, in log-return units. `seasonal_cost()` builds a broker-agnostic
  profile from any spread series.

## Targets (forward, over bars t+1..t+h; default h = 3, 6, 12, 18, 24)

| Axis | Target |
|---|---|
| direction | log displacement close[t+h] / close[t] |
| path | continuation: sign of the trailing move over a separate lookback × the forward displacement |
| volatility | log forward realised vol over t+1..t+h |

A window that touches an excluded span is dropped. By default that is the New York 16:00–20:00 roll, where FX
5m bars carry spread artefacts.

Horizons and the path lookback count **bars**, and the defaults are sized for 5m bars. On slower bars, pass
clock-equivalent values (on 1h bars, e.g. `horizons=(3, 6, 12)`, `lookbacks=(3,)`). A target with no measurable
out-of-sample bar is dropped with a warning (also listed in `meta["warnings"]` and the story), and an axis left with
no target gets no verdict. On 1h bars, a span longer than 20 bars always crosses the 4-hour roll window.

## Layers: what "beyond" means

| Layer | Baseline removed | Null |
|---|---|---|
| L0 raw | nothing | bar-shift rotation |
| L1 beyond clock | hour-of-week means on the New York 17:00 trading day | trading-day lattice rotation |
| L2 beyond HAR (volatility, primary) | a flexible HAR: 1h / 4h / 1D log RV, linear terms plus train-decile steps, with hour-of-week fixed effects | trading-day lattice rotation |
| L1v vol-standardized (direction, path) | the target ÷ exp(HAR forecast) | descriptive only |
| tails (volatility) | the rate of exceeding the HAR-conditional q95 / q99 / q05 / q01 | trading-day lattice rotation |

Every baseline is refitted per fold, on train data only, with a per-horizon purge.

## Inference

- **Walk-forward:** expanding train, quarterly test folds; a short final fold (< 25% of the median) is merged
  into the previous one.
- **Null:** rotating the label against the target keeps its run lengths, and the target's own overlap and
  persistence. Whole-trading-day offsets keep time of day.
- **Studentized statistic:** rotation is exact only for independence. Under "no information beyond the baseline",
  a label can line up with leftover heteroskedasticity, and the raw-mean null comes out too narrow. So the
  statistic is a one-vs-rest t with 5-trading-day cluster standard errors; the omnibus is Σt². η² is reported as
  the effect size only.
- **Multiple testing:** Holm across axis × horizon (the primary family), and separately across the tail family.
- **Within-condition test:** only (condition stratum, label state) cells with at least `min_bars` measurable bars of
  the observed label enter the sum of t², and the rotations use the same cells. A thin cell, with a handful of bars
  in one cluster, has a collapsed standard error and would otherwise dominate the statistic. `within_states` reports
  each cell's `n` and whether it was `tested`.
- **Bands:** moving-block bootstrap over trading days (5-day blocks). Quantile bands use per-day histograms.
  - The bands hold the label fixed, so they understate slow persistence.
  - When a band excludes zero but the rotation null does not reject, the null decides, and the story says so.

## Verdicts

- Per state: `flat`, `unstable`, `separable`, `actionable` or `insufficient`.
- Per label and axis: `KEEP`, `SEPARABLE-ONLY` or `FLAT`. `FLAT` ("no personality") is a classification outcome,
  not a failure.
- `actionable`, for direction and path: the 90% one-sided block-bootstrap bound clears cost. For volatility: the
  ratio bound is ≥ 1.25 or ≤ 0.80.
- Stability: the effect has the same sign in ≥ 2/3 of folds, and no single fold carries more than max(0.5, 2/F) of
  it, for F folds.
- State minimums scale with the out-of-sample size: bars ≥ max(500, 2% of bars), and episodes ≥ max(30, 2% of
  episodes).

## Validation (synthetic FX-like 5m data, 2 years, walk-forward from year 1)

| Check | Result |
|---|---|
| V1 size: independent label | 1–6% per test; family-wise 2–3% after Holm |
| V2 clock control: noisy London-hours label | L0 rejects volatility; L1 and L2 at size |
| V3 persistence control: trailing-RV tercile label | L2 volatility 7.0% (h=3) / 8.6% (h=24) |
| V4 power, K=2, dwell 48 | drift ±0.02σ: 78–98%; vol ×1.1: 40–100%; AR ±0.1: 48% at h=3 |
| Tail test | V1 1–7.3%, V3 1–8.3% |
| V5 alignment canary, V6 provenance guard | unit tests |

The validation code is in signalfoundry-lab `research/forward_separability/run_validation.py`; `synthetic.py`
generates the data.

## Known limits

- The longest horizon (h=24) runs mildly liberal, about 1.5–2× nominal. Treat significance carried only by it as
  weak.
- With HAR windows 1h / 4h / 1D, leftover volatility keeps multi-day persistence, so a persistent label faces a
  wide null. A 1-week HAR component is the next lever; adding it needs V3 re-run.
- The smallest attainable rotation p is 1 / (1 + B). With few rotations and a large Holm family, a family can be
  unresolvable; the story says so.
