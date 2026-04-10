# This implementation was adapted from Jonathan Shore's AmplitudeBasedLabeler.pyx
# (https://github.com/tr8dr/tseries-patterns/blob/master/tseries_patterns/labelers/AmplitudeBasedLabeler.pyx)
# Licensed under the MIT License.

# Copyright (c) 2015 Jonathan Shore
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from enum import Enum
from itertools import product
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from okmich_quant_labelling.utils.label_eval_util import trend_label_statistics


class PriceType(Enum):
    PRICE = 1
    CUMBPS = 2
    CUMR = 3

    def toBps(self, prices, scale=1e4):
        # 1st get into numpy array form
        if isinstance(prices, np.ndarray):
            pass
        elif isinstance(prices, pd.Series):
            prices = np.array(prices.values)
        else:
            prices = np.array(prices)

        if self.value == 2:
            return prices
        elif self.value == 1:
            return np.log(prices / prices[0]) * scale
        else:
            return prices * scale


class AmplitudeBasedLabeler:
    """
    Labels upward and downward momentum (or trend) movements where the following criteria are observed:
    - movement amplitude > minamp (usually defined in bps)
    - movement makes a new high (low) within Tinactive samples (where samples are # of bars)
    - movement is not broken by a move of minamp in the opposite direction

    These simple rules, together with a least squares filtration method, work astonishingly well in accurately
    identifying moves.

    It is recommended that prices be converted into cumulative returns, such that `minamp` can be defined
    from a return (or basis point) perspective.  Usually this would be accomplished by applying the following
    to a price series (in this case converting into cumulative bps):

    cumr = np.log(prices / prices[0]) * 1e4

    Amplitude `minamp` can then be defined as, say, 25bps instead of amplitude in price terms.
    """

    def __init__(self, minamp, Tinactive):
        """
        Label upward and downward momentum (or trend) movements

        :param minamp: minimum amplitude of move (usually in bps)
        :param Tinactive: maximum inactive period where no new high (low) achieved (unit: # of samples)
        """
        self.minamp = minamp
        self.Tinactive = Tinactive
        self.df = None

    def label(self, df, price_col="close", scale=1e4) -> pd.Series:
        """
        Perform labeling

        :param df: vector of bars, prices, or cumulative returns
        :param price_col: the name of the column of interest in the dataframe. Defaults to 'close'
        :param scale: bps scale
        :return: labels for the series
        """
        prices = df[price_col]

        cumr = PriceType.PRICE.toBps(prices, scale=scale)
        n = cumr.shape[0]
        labels = np.zeros(n).astype(np.double)

        self._pass1(cumr, labels)
        self._filter(cumr, labels)
        return pd.Series(labels, index=df.index)

    def _pass1(self, cumr, labels):
        """
        Brute-force labeling according to minamp and Tinactive rules.  This needs to be further filtered with
        OLS pass

        This code is ugly due to restrictions imposed by cython in terms of variable pre-declaration, etc.
        """

        length = cumr.shape[0]
        if length == 0:
            return

        Istart = 0
        Icursor = 0

        Imin = 0
        Imax = 0

        Vmin = cumr[0]
        Vmax = cumr[0]
        Vprior = cumr[0]

        v = 0.0

        while Icursor < length:
            v = cumr[Icursor]

            # determine whether there has been a retracement, requiring a split
            if (
                (Vmax - Vmin) >= self.minamp
                and Imin > Imax
                and (v - Vmin) >= self.minamp
            ):
                self._apply_label(labels, Istart, Imax - 1, 0.0)
                self._apply_label(labels, Imax, Imin, -1.0)
                Istart = Imin
                Imax = Icursor
                Vmax = v
            elif (
                (Vmax - Vmin) >= self.minamp
                and Imax > Imin
                and (Vmax - v) >= self.minamp
            ):
                self._apply_label(labels, Istart, Imin - 1, 0.0)
                self._apply_label(labels, Imin, Imax, +1.0)
                Istart = Imax
                Imin = Icursor
                Vmin = v

            # check for "inactive" period where price has not progressed since latest min/max (upward direction)
            elif Imax > Imin and (Icursor - Imax) >= self.Tinactive and v <= Vmax:
                if (Vmax - Vmin) >= self.minamp:
                    self._apply_label(labels, Istart, Imin - 1, 0.0)
                    self._apply_label(labels, Imin, Imax, +1.0)
                    self._apply_label(labels, Imax + 1, Icursor, 0.0)
                else:
                    self._apply_label(labels, Istart, Icursor, 0.0)

                Istart = Icursor
                Imax = Icursor
                Imin = Icursor
                Vmax = v
                Vmin = v

            # check for "inactive" period where price has not progressed since latest min/max (downward direction)
            elif Imin > Imax and (Icursor - Imin) >= self.Tinactive and v >= Vmin:
                if (Vmax - Vmin) >= self.minamp:
                    self._apply_label(labels, Istart, Imax - 1, 0.0)
                    self._apply_label(labels, Imax, Imin, -1.0)
                    self._apply_label(labels, Imin + 1, Icursor, 0.0)
                else:
                    self._apply_label(labels, Istart, Icursor, 0.0)

                Istart = Icursor
                Imax = Icursor
                Imin = Icursor
                Vmax = v
                Vmin = v

            # adjust local maximum
            if v >= Vmax:
                Imax = Icursor
                Vmax = v
            # adjust local minimum
            if v <= Vmin:
                Imin = Icursor
                Vmin = v

            Icursor += 1

        # finish end
        if (Vmax - Vmin) >= self.minamp and Imin > Imax:
            self._apply_label(labels, Istart, Imax - 1, 0.0)
            self._apply_label(labels, Imax, Imin, -1.0)
            self._apply_label(labels, Imin + 1, Icursor - 1, 0.0)

        elif (Vmax - Vmin) >= self.minamp and Imax > Imin:
            self._apply_label(labels, Istart, Imin - 1, 0.0)
            self._apply_label(labels, Imin, Imax, +1.0)
            self._apply_label(labels, Imax + 1, Icursor - 1, 0.0)
        else:
            self._apply_label(labels, Istart, Icursor - 1, 0.0)

    def _filter(self, cumr, labels):
        """
        Using distance from OLS regression, determine which points in a momentum region belong

        This code is ugly due to restrictions imposed by cython in terms of variable pre-declaration, etc.
        """

        length = cumr.shape[0]
        Ipos = 0
        Istart = 0
        Iend = 0

        Imaxfwd = 0
        Imaxback = 0
        Vmaxfwd = 0.0
        Vmaxback = 0.0

        fExy = 0.0
        fExx = 0.0
        fEx = 0.0
        fEy = 0.0

        bExy = 0.0
        bExx = 0.0
        bEx = 0.0
        bEy = 0.0

        beta = 0.0
        distance = 0.0

        Xc = 0.0
        Yc = 0.0
        dir = 0.0
        i = 0

        while Ipos < length:
            dir = labels[Ipos]
            if dir == 0.0:
                Ipos += 1
                continue

            # locate end of region
            Istart = Ipos
            Iend = Ipos
            while Iend < length and labels[Iend] == dir:
                Iend += 1
            Iend -= 1

            # setup for maximum extent
            Imaxfwd = Istart
            Imaxback = Iend
            Vmaxfwd = 0.0
            Vmaxback = 0.0

            # determine ols in the forward direction
            fExy = 0.0
            fExx = 0.0
            fEx = 0.0
            fEy = 0.0

            distance = 0.0
            for i in range(Istart, Iend + 1):
                Xc = float(i - Istart)
                Yc = cumr[i]
                fExy += Xc * Yc
                fExx += Xc * Xc
                fEx += Xc
                fEy += Yc

                if Xc > 0.0:
                    beta = (fExy - fEx * fEy / (Xc + 1.0)) / (
                        fExx - fEx * fEx / (Xc + 1.0)
                    )
                    distance = dir * beta * Xc

                if distance > Vmaxfwd:
                    Vmaxfwd = distance
                    Imaxfwd = i

            # determine ols in the backward direction
            bExy = 0.0
            bExx = 0.0
            bEx = 0.0
            bEy = 0.0

            distance = 0.0
            for i in range(Iend, Istart - 1, -1):
                Xc = float(Iend - i)
                Yc = cumr[i]
                bExy += Xc * Yc
                bExx += Xc * Xc
                bEx += Xc
                bEy += Yc

                if Xc > 0.0:
                    beta = (bExy - bEx * bEy / (Xc + 1.0)) / (
                        bExx - bEx * bEx / (Xc + 1.0)
                    )
                    distance = -dir * beta * Xc

                if distance > Vmaxback:
                    Vmaxback = distance
                    Imaxback = i

            # if neither direction meets required minimum, zero out
            if Vmaxfwd < self.minamp and Vmaxback < self.minamp:
                self._apply_label(labels, Istart, Iend, 0.0)
            else:
                # label forward region if meets size requirement
                if Vmaxfwd >= self.minamp:
                    self._apply_label(labels, Istart, Imaxfwd, dir)
                    self._apply_label(labels, Imaxfwd + 1, Imaxback - 1, 0.0)
                else:
                    self._apply_label(labels, Istart, Imaxback, 0.0)

                # label backward region if meets size requirement
                if Vmaxback >= self.minamp:
                    self._apply_label(labels, Imaxback, Iend, dir)
                else:
                    fwd_plus_1 = Imaxfwd + 1
                    maxvalue = Imaxback if Imaxback > fwd_plus_1 else fwd_plus_1
                    self._apply_label(labels, maxvalue, Iend, 0.0)

            Ipos = Iend + 1

    def _apply_label(self, labels, Istart, Iend, dir):
        """
        Apply a label value to a range of indices

        :param labels: label array to modify
        :param Istart: start index (inclusive)
        :param Iend: end index (inclusive)
        :param dir: direction label to apply
        """
        for i in range(Istart, Iend + 1):
            labels[i] = dir


def optimize_amplitude_base_labeler_parameters(df: pd.DataFrame, price_col: str = "close",
                                               min_amp_values: List[int] = None, t_inactive_values: List[float] = None,
                                               scale_value: List[float] = None, metric: str = "composite",
                                               verbose: bool = True,) -> Tuple[Dict, pd.DataFrame]:
    min_amp_values = min_amp_values or list(range(10, 150, 5))
    t_inactive_values = t_inactive_values or list(range(5, 150, 5))
    scale_value = scale_value or [1e3, 1e4, 1e5]

    results = []
    param_combinations = list(product(min_amp_values, t_inactive_values, scale_value))

    total = len(param_combinations)
    for idx, (min_amp, Tinactive, scale) in enumerate(param_combinations):
        if verbose and idx % 200 == 0:
            print(f"Testing combination {idx + 1}/{total}...")

        try:
            # Generate labels
            labeler = AmplitudeBasedLabeler(minamp=min_amp, Tinactive=Tinactive)

            # Calculate forward returns
            df_eval = df.copy()
            df_eval["label"] = labeler.label(df_eval, price_col=price_col, scale=scale)
            df_eval["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
            df_eval = df_eval.dropna()

            # Get statistics using your function
            stats = trend_label_statistics(
                df_eval, state_col="label", return_col="log_return"
            )

            # Skip if we don't have all three labels
            if len(stats) < 3:
                continue

            # Extract key metrics for each label
            long_stats = (
                stats[stats["label"] == 1].iloc[0]
                if 1 in stats["label"].values
                else None
            )
            short_stats = (
                stats[stats["label"] == -1].iloc[0]
                if -1 in stats["label"].values
                else None
            )
            neutral_stats = (
                stats[stats["label"] == 0].iloc[0]
                if 0 in stats["label"].values
                else None
            )

            if long_stats is None or short_stats is None:
                continue

            # Core metrics
            mean_ret_long = long_stats["mean"]
            mean_ret_short = short_stats["mean"]
            mean_ret_neutral = neutral_stats["mean"] if neutral_stats is not None else 0

            sum_ret_long = long_stats["sum"]
            sum_ret_short = short_stats["sum"]

            std_ret_long = long_stats["std"]
            std_ret_short = short_stats["std"]
            std_all = df_eval["log_return"].std()

            count_long = long_stats["count"]
            count_short = short_stats["count"]

            min_ret_long = long_stats["min"]
            min_ret_short = short_stats["min"]
            max_ret_long = long_stats["max"]
            max_ret_short = short_stats["max"]

            # Calculate various optimization metrics ##############
            # 1. Return Separation (Sharpe-like): How well separated are long vs short?
            return_separation = (mean_ret_long - mean_ret_short) / (std_all + 1e-9)

            # 2. Total Return: Sum of all profitable trades
            total_return = sum_ret_long + abs(sum_ret_short)

            # 3. Win Rate: Percentage of positive returns for each label
            win_rate_long = (df_eval[df_eval["label"] == 1]["log_return"] > 0).mean()
            win_rate_short = (df_eval[df_eval["label"] == -1]["log_return"] < 0).mean()
            avg_win_rate = (win_rate_long + win_rate_short) / 2

            # 4. Sharpe Ratio for each signal type
            sharpe_long = (
                mean_ret_long / (std_ret_long + 1e-9) if std_ret_long > 0 else 0
            )
            sharpe_short = (
                abs(mean_ret_short) / (std_ret_short + 1e-9) if std_ret_short > 0 else 0
            )
            avg_sharpe = (sharpe_long + sharpe_short) / 2

            # 5. Risk-Adjusted Return: Consider worst-case scenarios
            risk_adj_long = (
                mean_ret_long / (abs(min_ret_long) + 1e-9)
                if min_ret_long < 0
                else mean_ret_long
            )
            risk_adj_short = (
                abs(mean_ret_short) / (abs(max_ret_short) + 1e-9)
                if max_ret_short > 0
                else abs(mean_ret_short)
            )
            risk_adjusted = (risk_adj_long + risk_adj_short) / 2

            # 6. Consistency: Lower std relative to mean is better
            consistency_long = abs(mean_ret_long) / (std_ret_long + 1e-9)
            consistency_short = abs(mean_ret_short) / (std_ret_short + 1e-9)
            consistency = (consistency_long + consistency_short) / 2

            # 7. Balance: We want both long and short to work well
            balance_score = min(abs(mean_ret_long), abs(mean_ret_short)) / max(
                abs(mean_ret_long), abs(mean_ret_short)
            )

            persistence_long = long_stats["avg_persistence"] / 3.0
            persistence_short = short_stats["avg_persistence"] / 3.0
            avg_persistence = (persistence_long + persistence_short) / 2

            # 9. COMPOSITE SCORE - Weighted combination of key metrics
            composite = (
                0.30 * return_separation  # Primary: separation
                + 0.25 * (avg_sharpe / 2)  # Risk-adjusted returns (normalized)
                + 0.20 * (avg_win_rate - 0.5) * 10  # Win rate above 50% (scaled)
                + 0.10 * consistency / 5  # Consistency (scaled)
                + 0.10 * balance_score  # Balance between long/short
                + 0.05
                * min(avg_persistence, 3.0)  # Persistence bonus (capped at 3x baseline)
            )

            # Choose optimization metric
            metric_map = {
                "composite": composite,
                "return_separation": return_separation,
                "total_return": total_return,
                "win_rate": avg_win_rate,
                "sharpe_ratio": avg_sharpe,
                "risk_adjusted": risk_adjusted,
                "persistence": avg_persistence,
            }
            score = metric_map.get(metric, composite)

            results.append(
                {
                    "min_amp": min_amp,
                    "Tinactive": Tinactive,
                    "scale": scale,
                    "score": score,
                    "return_separation": return_separation,
                    "total_return": total_return,
                    "composite": composite,
                    "mean_ret_long": mean_ret_long,
                    "mean_ret_short": mean_ret_short,
                    "mean_ret_neutral": mean_ret_neutral,
                    "sum_ret_long": sum_ret_long,
                    "sum_ret_short": sum_ret_short,
                    "sharpe_long": sharpe_long,
                    "sharpe_short": sharpe_short,
                    "avg_sharpe": avg_sharpe,
                    "win_rate_long": win_rate_long,
                    "win_rate_short": win_rate_short,
                    "avg_win_rate": avg_win_rate,
                    "consistency": consistency,
                    "balance_score": balance_score,
                    "risk_adjusted": risk_adjusted,
                    "persistence_long": persistence_long,
                    "persistence_short": persistence_short,
                    "avg_persistence": avg_persistence,
                    "count_long": count_long,
                    "count_short": count_short,
                    "std_ret_long": std_ret_long,
                    "std_ret_short": std_ret_short,
                    "avg_persistence_long": long_stats["avg_persistence"],
                    "max_persistence_long": long_stats["max_persistence"],
                    "avg_persistence_short": short_stats["avg_persistence"],
                    "max_persistence_short": short_stats["max_persistence"],
                }
            )

        except Exception as e:
            if verbose:
                print(f"Error with params {min_amp}/{Tinactive}/{scale}: {e}")
            continue

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)

    if len(results_df) == 0:
        raise ValueError("No valid parameter combinations found!")

    if len(results_df) == 0:
        raise ValueError("No valid parameter combinations found!")

    best_params = results_df.iloc[0, :4].to_dict()
    return best_params, results_df
