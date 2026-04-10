import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import plotly.offline as py
from prophet import Prophet
from prophet.plot import plot_components_plotly
from prophet.serialize import model_from_json, model_to_json


# For Inference Service only
class ProphetFeatureGenerationService:
    def __init__(self, model_path):
        self.processor = ProphetFeatures.load(model_path)

    def get_features(self, new_data):
        return self.processor.predict(new_data)


class ProphetFeatures:

    def __init__(
        self,
        target_col="close",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        add_country_holidays=None,
        extra_regressors=None,
        resample_agg_func="sum",
        is_input_lower_freq=True,
        keep_prediction_freq=False,
        **prophet_kwargs,
    ):
        self.target_col = target_col
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.add_country_holidays = add_country_holidays
        self.extra_regressors = extra_regressors or []
        self.resample_agg_func = resample_agg_func
        self.is_input_lower_freq = is_input_lower_freq  # is the timeframe of the series going to be below hourly
        self.keep_prediction_freq = keep_prediction_freq  # return the hourly frequency even if the input is lower
        self.prophet_kwargs = prophet_kwargs

        self.model = None
        self.forecast_hourly = None
        self.features = None
        self.hourly_data = None
        self.is_fitted = False
        self.fitted_date_range = None
        self.model_version = "1.0"
        self.rename_features_map = {
            "trend": "trend_component",
            "daily_cycle": "daily_seasonality",
            "weekly_cycle": "weekly_seasonality",
            "holidays": "holiday_effect",
            "additive_terms": "total_seasonal_effect",
            "residual": "residual",
        }

    def _initialize_model(self):
        """Initializes and configures the Prophet model."""
        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=self.yearly_seasonality,
            **self.prophet_kwargs,
        )

        if self.daily_seasonality:
            self.model.add_seasonality(name="daily_cycle", period=1, fourier_order=12)

        if self.weekly_seasonality:
            self.model.add_seasonality(name="weekly_cycle", period=7, fourier_order=8)

        if self.add_country_holidays:
            self.model.add_country_holidays(country_name=self.add_country_holidays)

        for regressor in self.extra_regressors:
            self.model.add_regressor(regressor)
        return self.model

    def fit(self, df) -> "ProphetFeatures":
        data = self._resample_to_hourly(df) if self.is_input_lower_freq else df
        if self.is_input_lower_freq:
            self.hourly_data = data
        self._initialize_model()
        self.model.fit(data)
        self.forecast_hourly = self.model.predict(data)

        if self.is_input_lower_freq:
            self._extract_and_broadcast_features(data, df)

        self.is_fitted = True
        self.fitted_date_range = {"start": data["ds"].min(), "end": data["ds"].max()}
        return self

    def predict(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Pure inference: generates features for possibly lower timeframe data without refitting.

        Parameters:
            df_new (pd.DataFrame): New possibly lower timeframe data for inference

        Returns:
            pd.DataFrame: Prophet features for the new data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before inference. Call fit() first.")

        # Resample new data if needed to hourly
        data_new = (
            self._resample_to_hourly(df_new) if self.is_input_lower_freq else df_new
        )

        # Generate forecast for new hourly data
        forecast_new = self.model.predict(data_new)

        # Extract features from the forecast
        feature_columns = ["ds", "trend"]
        if self.daily_seasonality and "daily_cycle" in forecast_new.columns:
            feature_columns.append("daily_cycle")
        if self.weekly_seasonality and "weekly_cycle" in forecast_new.columns:
            feature_columns.append("weekly_cycle")
        if "holidays" in forecast_new.columns:
            feature_columns.append("holidays")
        feature_columns.append("additive_terms")

        features_new = forecast_new[feature_columns].copy()
        # Calculate residuals if we have actual values
        if "y" in data_new.columns:
            features_new["residual"] = (
                data_new["y"].values - forecast_new["yhat"].values
            )
        else:
            features_new["residual"] = np.nan  # For pure inference without actual

        return self._broadcast_to_lower_tf(
            features_new.rename(columns=self.rename_features_map), df_new
        )

    def _resample_to_hourly(self, sub_hourly_df):
        """Resamples data to hourly frequency."""
        df_resample = sub_hourly_df.copy()

        resample_cols = [self.target_col] + self.extra_regressors
        hourly_data = (
            df_resample[self.target_col]
            .resample("1h")
            .agg(self.resample_agg_func)
            .to_frame()
        )
        for col in resample_cols:
            if col != self.target_col:
                hourly_data[col] = df_resample[col].resample("1h").mean()

        hourly_data["ds"] = hourly_data.index
        hourly_data = hourly_data.rename({self.target_col: "y"}, axis=1)
        return hourly_data.dropna()

    def _extract_and_broadcast_features(self, resampled_df, sub_hourly_df):
        feature_columns = ["ds", "trend"]

        if self.daily_seasonality and "daily_cycle" in self.forecast_hourly.columns:
            feature_columns.append("daily_cycle")

        if self.weekly_seasonality and "weekly_cycle" in self.forecast_hourly.columns:
            feature_columns.append("weekly_cycle")

        if "holidays" in self.forecast_hourly.columns:
            feature_columns.append("holidays")

        feature_columns.append("additive_terms")

        hourly_features = self.forecast_hourly[feature_columns].copy()
        hourly_features["residual"] = (
            resampled_df["y"].values - self.forecast_hourly["yhat"].values
        )
        self.features = self._broadcast_to_lower_tf(
            hourly_features.rename(columns=self.rename_features_map), sub_hourly_df
        )

    def _broadcast_to_lower_tf(self, hourly_features, sub_hourly_df):
        """Broadcasts hourly features to lower frequency."""
        if self.keep_prediction_freq:
            hourly_features.rename(columns={"ds": "time"}, inplace=True)
            return hourly_features.set_index("time")

        features_ltf = pd.DataFrame(index=sub_hourly_df.index)
        features_ltf["merge_key"] = sub_hourly_df.index.floor("h")
        hourly_features["merge_key"] = hourly_features["ds"].dt.floor("h")

        features_ltf = features_ltf.merge(
            hourly_features.drop(columns=["ds"]), on="merge_key", how="left"
        )
        features_ltf.index = sub_hourly_df.index

        features_ltf = features_ltf.drop(columns=["merge_key"])
        feature_cols = [col for col in features_ltf.columns]
        rename_dict = {col: f"prophet_{col}" for col in feature_cols}

        return features_ltf.rename(columns=rename_dict)

    def get_features(self):
        """Returns the generated features at the lower timeframe."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting features.")
        return self.features.copy()

    def save(self, save_path: str, overwrite: bool = False):
        """
        Saves the entire ProphetFeatures instance to disk.

        Parameters:
            save_path (str): Directory path to save the model
            overwrite (bool): Whether to overwrite existing files
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Check if files exist
        model_path = save_path / "prophet_model.json"
        state_path = save_path / "prophet_features_state.pkl"

        if not overwrite and (model_path.exists() or state_path.exists()):
            raise FileExistsError(
                "Model files already exist. Use overwrite=True to replace."
            )

        # Save Prophet model
        if self.model:
            with open(model_path, "w") as f:
                json.dump(model_to_json(self.model), f)

        # Save class state (excluding the Prophet model)
        state = {
            "target_col": self.target_col,
            "daily_seasonality": self.daily_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "yearly_seasonality": self.yearly_seasonality,
            "add_country_holidays": self.add_country_holidays,
            "extra_regressors": self.extra_regressors,
            "resample_agg_func": self.resample_agg_func,
            "prophet_kwargs": self.prophet_kwargs,
            "is_input_lower_freq": self.is_input_lower_freq,
            "keep_prediction_freq": self.keep_prediction_freq,
            "is_fitted": self.is_fitted,
            "fitted_date_range": self.fitted_date_range,
            "model_version": self.model_version,
            "hourly_data_info": (
                {
                    "start": (
                        self.hourly_data["ds"].min()
                        if self.hourly_data is not None
                        else None
                    ),
                    "end": (
                        self.hourly_data["ds"].max()
                        if self.hourly_data is not None
                        else None
                    ),
                    "rows": (
                        len(self.hourly_data) if self.hourly_data is not None else 0
                    ),
                }
                if self.hourly_data is not None
                else None
            ),
        }

        with open(state_path, "wb") as f:
            pickle.dump(state, f)

        return str(save_path)

    @classmethod
    def load(cls, load_path: str):
        """
        Loads a previously saved ProphetFeatures instance.

        Parameters:
            load_path (str): Directory path where the model is saved

        Returns:
            ProphetFeatures: Loaded instance
        """
        load_path = Path(load_path)
        model_path = load_path / "prophet_model.json"
        state_path = load_path / "prophet_features_state.pkl"

        if not model_path.exists() or not state_path.exists():
            raise FileNotFoundError("Model files not found in the specified path")

        # Load class state
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        # Create new instance
        instance = cls(
            target_col=state["target_col"],
            daily_seasonality=state["daily_seasonality"],
            weekly_seasonality=state["weekly_seasonality"],
            yearly_seasonality=state["yearly_seasonality"],
            add_country_holidays=state["add_country_holidays"],
            extra_regressors=state["extra_regressors"],
            is_input_lower_freq=state["is_input_lower_freq"],
            keep_prediction_freq=state["keep_prediction_freq"],
            resample_agg_func=state["resample_agg_func"],
            **state["prophet_kwargs"],
        )

        # Load Prophet model
        if state["is_fitted"]:
            with open(model_path, "r") as f:
                model_json = json.load(f)
            instance.model = model_from_json(model_json)
            instance.is_fitted = state["is_fitted"]
            instance.fitted_date_range = state["fitted_date_range"]
            instance.model_version = state.get("model_version", "1.0")

        return instance

    def get_model_info(self) -> Dict[str, Any]:
        """Returns comprehensive information about the fitted model."""
        if not self.is_fitted:
            return {"status": "not_fitted"}

        info = {
            "status": "fitted",
            "model_version": self.model_version,
            "target_column": self.target_col,
            "is_input_lower_freq": self.is_input_lower_freq,
            "keep_prediction_freq": self.keep_prediction_freq,
            "fitted_date_range": self.fitted_date_range,
            "training_hours": (
                len(self.hourly_data) if self.hourly_data is not None else 0
            ),
            "seasonalities": {
                "daily": self.daily_seasonality,
                "weekly": self.weekly_seasonality,
                "yearly": self.yearly_seasonality,
            },
            "extra_regressors": self.extra_regressors,
            "resample_aggregation": self.resample_agg_func,
        }
        return info

    def plot_components(self, figsize=(1200, 600)):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting.")
        if self.forecast_hourly is None:
            raise ValueError(
                "forecast_hourly must be set before plotting. Fit the model again before plotting."
            )

        fig = plot_components_plotly(self.model, self.forecast_hourly)
        fig.update_layout(width=figsize[0], height=figsize[1])
        py.iplot(fig)


def create_facebook_prophet_features(
    df, target_col="tick_volume", add_yearly_seasonality: bool = False
) -> Tuple[pd.DataFrame, ProphetFeatures]:
    """
    Quick feature generation based on ProphetFeatures.

    :param df:
    :param target_col:
    :param add_yearly_seasonality:
    :return:
    """
    prophet_feature = ProphetFeatures(
        target_col=target_col,
        is_input_lower_freq=True,
        keep_prediction_freq=True,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=add_yearly_seasonality,
        resample_agg_func="sum",
    ).fit(df)
    predict_df = prophet_feature.predict(df)
    output_cols = [
        "daily_seasonality",
        "weekly_seasonality",
        "total_seasonal_effect",
        "residual",
    ]
    if add_yearly_seasonality:
        output_cols.append("yearly_seasonality")
    output_df = pd.merge_asof(df, predict_df, left_index=True, right_index=True)[
        output_cols
    ]
    return output_df, prophet_feature
