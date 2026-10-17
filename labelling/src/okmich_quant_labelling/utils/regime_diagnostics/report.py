"""The diagnostic report (one HTML) for a RegimeDiagnostics: story first, distributions next, verdicts last.

Sections: 0 story · 1 the label · 2 descriptive (inside episodes, train vs out of sample) · 3 forward distributions
(quantile tables and shifts with bands, raw / vol-standardized / beyond the baseline) · 4 volatility tails ·
5 conditional on the other regime dimension (expectation table, within-condition test) · 6 verdicts · appendix nulls.
"""
from __future__ import annotations

from html import escape

import pandas as pd

from .core import PRIMARY_LAYER

DESCRIPTIVE_METRICS = [
    "volatility", "mean_return", "return_skewness", "return_kurtosis", "autocorrelation_lag1", "efficiency_ratio",
    "efficiency_ratio_rw", "trend_persistence", "volatility_stability", "max_drawdown", "regime_discriminability",
    "discriminability_metric",
]
DURATION_METRICS = ["n_observations", "frequency_pct", "min_duration", "q25_duration", "median_duration",
                    "mean_duration", "q75_duration", "max_duration", "regime_stability"]
BAND_LEVELS = (0.05, 0.5, 0.95)

_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; color: #0b0b0b; background: #fcfcfb; margin: 0; }
main { max-width: 1400px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 40px 0 8px; border-top: 1px solid #e6e5e1; padding-top: 20px; }
h3 { font-size: 15px; margin: 24px 0 6px; color: #52514e; }
p, li { color: #52514e; line-height: 1.5; max-width: 1000px; } .stamp { color: #b3261e; font-weight: 600; }
.meta { font-size: 13px; color: #52514e; }
.story { white-space: pre-wrap; background: #f3f2ef; border-radius: 6px; padding: 14px 16px; font-size: 13.5px;
line-height: 1.55; max-width: 1100px; }
table { border-collapse: collapse; font-size: 12.5px; margin: 8px 0 16px; }
th, td { border-bottom: 1px solid #e6e5e1; padding: 4px 10px; text-align: right; white-space: nowrap; }
th { color: #52514e; font-weight: 600; background: #f3f2ef; } td:first-child, th:first-child { text-align: left; }
.scroll { overflow-x: auto; }
"""


def _table(df: pd.DataFrame, index: bool = True) -> str:
    if df is None or df.empty:
        return "<p>(not available)</p>"
    html = df.to_html(index=index, na_rep="", float_format=lambda x: f"{x:.4g}", border=0)
    return f'<div class="scroll">{html}</div>'


def _descriptive_side_by_side(train: pd.DataFrame, oos: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    parts = {}
    for span, df in (("train", train), ("out-of-sample", oos)):
        if df.empty:
            continue
        parts[span] = df.set_index("regime")[[m for m in metrics if m in df.columns]].T
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, axis=1)
    combined.columns = [f"state {state} · {span}" for span, state in combined.columns]
    return combined[sorted(combined.columns, key=lambda c: (c.split(" · ")[0], c.split(" · ")[1] != "train"))]


def _quantile_tables(forward) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw per-state quantiles, and primary-layer quantile shifts with their 90% bands, one row per axis x h x state."""
    q = forward.quantiles
    raw = q[q["layer"] == "L0_raw"]
    raw_table = raw.pivot_table(index=["axis", "horizon", "state"], columns="quantile", values="value", aggfunc="first")
    raw_table.columns = [f"q{int(round(c * 100)):02d}" for c in raw_table.columns]

    primary_layers = {str(axis): str(layer) for axis, layer in PRIMARY_LAYER.items()}
    prim = q[(q["layer"] == q["axis"].map(primary_layers)) & (q["state"].astype(str) != "all")
             & q["quantile"].isin(BAND_LEVELS)].copy()
    prim["cell"] = [f"{s:+.3g} [{lo:+.3g}, {hi:+.3g}]"
                    for s, lo, hi in zip(prim["shift"], prim["shift_lower"], prim["shift_upper"])]
    shift_table = prim.pivot_table(index=["axis", "horizon", "state"], columns="quantile", values="cell",
                                   aggfunc="first")
    shift_table.columns = [f"shift q{int(round(c * 100)):02d} [90% band]" for c in shift_table.columns]
    return raw_table, shift_table


def _conditional_tables(conditional) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if conditional is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    exp = conditional.expectations
    shown = exp[exp["layer"].isin(["raw", "volstd"])].copy()
    for tag in ("mean", "q05", "q50", "q95"):
        lo, hi = ("mean_lower", "mean_upper") if tag == "mean" else (f"{tag}_lower", f"{tag}_upper")
        if tag in shown.columns:
            shown[tag] = [f"{v:.4g} [{a:.4g}, {b:.4g}]" if pd.notna(v) else ""
                          for v, a, b in zip(shown[tag], shown[lo], shown[hi])]
    columns = ["target", "layer", "condition", "state", "n", "episodes", "enough", "mean", "q05", "q50", "q95"]
    return (shown[[c for c in columns if c in shown.columns]].reset_index(drop=True), conditional.within,
            conditional.within_states)


def write_report(diag, charts: dict, path) -> None:
    forward = diag.forward
    meta = forward.meta
    figures, first = {}, True
    for key, fig in charts.items():
        figures[key] = fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False)
        first = False

    raw_q, shift_q = _quantile_tables(forward)
    expectations, within, within_states = _conditional_tables(diag.conditional)
    s = forward.summary
    verdict_summary = s[s["primary"]][["axis", "horizon", "lookback", "layer", "eta2", "statistic", "B", "p",
                                       "p_holm"]].reset_index(drop=True)
    state_columns = ["axis", "horizon", "state", "n_bars", "n_episodes", "primary_effect", "t_state", "p_state_holm",
                     "consistency", "concentration", "action_value", "action_lower", "action_upper", "verdict"]
    per_state = forward.per_state[[c for c in state_columns if c in forward.per_state.columns]]
    stamp = ('<p class="stamp">DIAGNOSTIC — not tradeable (label fitted on out-of-sample data, or diagnostic '
             'mode)</p>') if forward.diagnostic else ""
    oos_index = forward.values.index
    conditional_html = ""
    if diag.conditional is not None:
        conditional_html = f"""<h2>5 · Conditional on {escape(diag.condition_name)}</h2>
<p>Expectations per label state × condition cell (out of sample; 90% day-block bands). Cells under the minimum are
marked <code>enough = False</code> and should not be read.</p>
{_table(expectations, index=False)}
<h3>Does the label still separate inside each condition stratum? (label rotated, condition held fixed)</h3>
{_table(within, index=False)}
{_table(within_states, index=False)}"""

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Regime diagnostics — {escape(diag.name)}</title>
<style>{_CSS}</style></head><body><main>
<h1>Regime diagnostics — {escape(diag.name)}</h1>
<p class="meta">Out of sample {oos_index[0]:%Y-%m-%d} → {oos_index[-1]:%Y-%m-%d} ·
{meta['n_folds']} walk-forward folds · states {escape(str(meta['states']))} ·
mean dwell {meta['mean_dwell']:.1f} bars · {meta['B_day_rotations']} day rotations ·
Holm family {meta['family_size']}</p>{stamp}

<h2>0 · The story</h2>
<div class="story">{escape(diag.story)}</div>

<h2>1 · The label</h2>
<p>Occupancy and episode lengths. A state that lives shorter than the holding horizon is a feature-level state,
not a regime a trade can be held through.</p>
{_table(_descriptive_side_by_side(diag.descriptive_train, diag.descriptive_oos, DURATION_METRICS))}
{figures['states_on_price']}

<h2>2 · Descriptive — inside the episodes</h2>
<p>What each state looks like while you are in it (<code>label_path_structure_statistics</code>), train and
out-of-sample spans side by side: a profile that changes between spans is itself a warning.</p>
{_table(_descriptive_side_by_side(diag.descriptive_train, diag.descriptive_oos, DESCRIPTIVE_METRICS))}

<h2>3 · Forward — what follows the decision bar</h2>
<h3>Per-state quantiles of the raw forward target</h3>
{_table(raw_q)}
<h3>Quantile shift vs all bars beyond the baseline, 90% day-block band</h3>
<p>A band that excludes zero at q50 moves the centre; one only at q05 or q95 lives in a tail.</p>
{_table(shift_q)}
{figures['quantile_profile_primary']}
{figures['quantile_profile_volstd']}
{figures['distribution_curves_volstd']}
{figures['quantile_profile_raw']}
{figures['fold_effects']}
{figures['robustness']}
{figures['distributions_by_fold']}

<h2>4 · Volatility tails beyond the HAR-conditional distribution</h2>
<p>Rate of forward-vol tail events (beyond the train q95 / q99, or below q05 / q01, of the HAR residual within the
forecast's decile) per state vs the rest. Studentized rotation tests, Holm over the tail family.</p>
{_table(forward.tails, index=False)}
{conditional_html}

<h2>6 · Verdicts</h2>
<p>{", ".join(f"{axis}: <b>{verdict}</b>" for axis, verdict in forward.label_verdicts.items())}. Verdicts summarise
the distributions above; they do not replace them.</p>
{_table(verdict_summary, index=False)}
{_table(per_state, index=False)}

<h2>Appendix · Rotation nulls</h2>
{figures['null']}
</main></body></html>"""
    path.write_text(html, encoding="utf-8")
