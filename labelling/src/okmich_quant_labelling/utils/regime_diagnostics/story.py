"""tell_story(): a plain-language read of a regime diagnostic, generated only from its tables.

Every sentence carries a number and, where one exists, its band; nothing is claimed beyond the tables. The story
leads with what the label is and what follows each state, and ends with the verdicts and caveats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 0.05


def _pct(log_shift: float) -> str:
    return f"{(np.exp(log_shift) - 1) * 100:+.1f}%"


def _band(lo: float, hi: float, fmt) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]" if np.isfinite(lo) and np.isfinite(hi) else ""


def _middle_horizon(forward) -> int:
    horizons = sorted(forward.summary["horizon"].unique())
    return int(horizons[len(horizons) // 2])


def _overview(name, forward, descriptive_oos) -> list[str]:
    meta = forward.meta
    index = forward.values.index
    text = [f"{name}: {len(meta['states'])} states, out of sample {index[0]:%Y-%m-%d} to {index[-1]:%Y-%m-%d} over "
            f"{meta['n_folds']} walk-forward folds."]
    if not descriptive_oos.empty:
        parts = [f"state {r.regime} holds {r.frequency_pct:.0f}% of bars in episodes of median {r.median_duration:.0f} "
                 f"bars (mean {r.mean_duration:.0f})" for r in descriptive_oos.itertuples()]
        text.append("Occupancy: " + "; ".join(parts) + ".")
        shortest = min(forward.summary["horizon"])
        if descriptive_oos["median_duration"].median() < shortest:
            text.append(f"Most episodes end before the shortest horizon ({shortest} bars): this is a feature-level "
                        "state, not a regime a trade can be held through.")
    if forward.diagnostic:
        text.append("DIAGNOSTIC — the label was fitted on out-of-sample data; nothing here is tradeable.")
    return text


def _volatility(forward, h) -> list[str]:
    q = forward.quantiles
    part = q[(q["axis"] == "volatility") & (q["horizon"] == h) & (q["layer"] == "L2_beyond_har")
             & (q["state"].astype(str) != "all")]
    if part.empty:
        return []
    text = [f"Forward volatility over the next {h} bars, beyond the HAR forecast (trailing vol and the clock):"]
    starred = False
    for state, rows in part.groupby("state", sort=False):
        cells = []
        for level in (0.5, 0.95, 0.99):
            r = rows[rows["quantile"] == level]
            if r.empty:
                continue
            r = r.iloc[0]
            excludes = r["shift_lower"] > 0 or r["shift_upper"] < 0
            starred |= excludes
            cells.append(f"q{int(level * 100)} {_pct(r['shift'])} {_band(r['shift_lower'], r['shift_upper'], _pct)}"
                         + (" *" if excludes else ""))
        text.append(f"  after state {state}: " + ", ".join(cells))
    text.append("  (* = 90% day-block band excludes zero; bands are pointwise, so trust patterns that repeat across "
                "horizons.)")
    s = forward.summary
    test = s[(s["axis"] == "volatility") & (s["horizon"] == h) & s["primary"]]
    if starred and not test.empty and test["p"].iloc[0] >= ALPHA:
        text.append(f"  But the rotation null does not reject at h={h} (p {test['p'].iloc[0]:.3f}): a random label "
                    "with this persistence shows shifts this large. The day-block bands hold the label fixed and "
                    "understate slow volatility persistence; the null decides.")
    return text


def _resolvable(rotations: int, family: int) -> bool:
    """Whether the smallest attainable rotation p, 1 / (1 + B), can clear Holm's first step alpha / family."""
    return family == 0 or 1 / (1 + rotations) < ALPHA / family


def _tails(forward) -> list[str]:
    tails = forward.tails
    if tails is None or tails.empty:
        return []
    hits = tails[(tails["p_omnibus_holm"] < ALPHA) & (tails["p_state_holm"] < ALPHA)]
    if len(forward.meta["states"]) == 2:
        hits = hits[hits["rate"] > hits["rate_rest"]]  # the other state's row is its mirror image
    if hits.empty:
        family = tails.drop_duplicates(["target", "tail"]).shape[0]
        if not _resolvable(forward.meta["B_day_rotations"], family):
            rotations = forward.meta["B_day_rotations"]
            return [f"Volatility tails: not resolvable — {rotations} rotations cannot clear Holm over {family} tail "
                    "tests; read the tails table, not this line."]
        return ["Volatility tails: no state changes the rate of tail events beyond the HAR-conditional distribution "
                "(Holm over the tail family)."]
    text = ["Volatility tails beyond the HAR-conditional distribution (Holm over the tail family):"]
    for r in hits.sort_values(["horizon", "tail"]).itertuples():
        text.append(f"  h={r.horizon} {r.side} {r.tail.split('_')[-1]}: state {r.state} {r.rate * 100:.2f}% vs "
                    f"{r.rate_rest * 100:.2f}% elsewhere (nominal {r.nominal_rate * 100:.0f}%)")
    return text


def _scaled_moves(forward, h) -> list[str]:
    q = forward.quantiles
    text = []
    for axis, what in (("direction", "forward move"), ("path", "continuation of the recent move")):
        part = q[(q["axis"] == axis) & (q["horizon"] == h) & (q["layer"] == "L1v_vol_standardized")]
        if part.empty:
            continue
        pooled = part[part["state"].astype(str) == "all"].set_index("quantile")["value"]
        pieces = []
        for state, rows in part[part["state"].astype(str) != "all"].groupby("state", sort=False):
            v = rows.set_index("quantile")["value"]
            pieces.append(f"state {state} median {v.get(0.5, np.nan):+.3f}, q05..q95 {v.get(0.05, np.nan):+.2f}.."
                          f"{v.get(0.95, np.nan):+.2f}")
        text.append(f"Vol-standardized {what} over {h} bars (units of forecast vol; all bars median "
                    f"{pooled.get(0.5, np.nan):+.3f}): " + "; ".join(pieces) + ".")
    return text


def _conditional(conditional, h) -> list[str]:
    if conditional is None:
        return []
    text = []
    within = conditional.within
    if not within.empty:
        hits = within[within["p_holm"] < ALPHA]
        if hits.empty and not _resolvable(int(within["B"].iloc[0]), len(within)):
            text.append(f"Within the condition strata: not resolvable — {int(within['B'].iloc[0])} rotations cannot "
                        f"clear Holm over {len(within)} targets; read the expectation table instead.")
        elif hits.empty:
            text.append("Within the condition strata, the label separates no forward target (Holm across targets): "
                        "inside each condition, its states look alike going forward.")
        else:
            text.append("Within the condition strata, the label still separates: "
                        + ", ".join(f"{r.target} (Holm p {r.p_holm:.3f})" for r in hits.itertuples()) + ".")
    exp = conditional.expectations
    vol = exp[(exp["target"] == f"volatility_h{h}") & (exp["layer"] == "raw") & exp["enough"]]
    if not vol.empty:
        text.append(f"Expected forward volatility over {h} bars by cell (median realised vol, 90% band):")
        for r in vol.itertuples():
            text.append(f"  {r.condition} | state {r.state}: {np.exp(r.q50) * 1e4:.1f} bp "
                        f"[{np.exp(r.q50_lower) * 1e4:.1f}, {np.exp(r.q50_upper) * 1e4:.1f}], "
                        f"q95 {np.exp(r.q95) * 1e4:.1f} bp (n={r.n}, {r.episodes} episodes)")
    return text


def _verdicts(forward) -> list[str]:
    s = forward.summary[forward.summary["primary"]]
    text = ["Verdicts (they summarise the distributions above): "
            + ", ".join(f"{axis} {verdict}" for axis, verdict in forward.label_verdicts.items()) + "."]
    for axis, rows in s.groupby("axis", sort=False):
        best = rows.loc[rows["p_holm"].idxmin()]
        text.append(f"  {axis}: best primary test h={best['horizon']}, Holm p {best['p_holm']:.3f}, "
                    f"eta2 {best['eta2']:.2e}.")
    text.append("Caveat: the longest horizon runs mildly liberal in validation; a result carried only by it is weak.")
    text.extend(f"Warning: {w}" for w in forward.meta["warnings"])
    return text


def tell_story(name: str, forward, descriptive_oos: pd.DataFrame, conditional=None) -> str:
    h = _middle_horizon(forward)
    sections = [_overview(name, forward, descriptive_oos), _volatility(forward, h), _tails(forward),
                _scaled_moves(forward, h), _conditional(conditional, h), _verdicts(forward)]
    return "\n\n".join("\n".join(lines) for lines in sections if lines)
