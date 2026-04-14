import os
from typing import List, Callable, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
from joblib import Parallel, delayed
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch, MeanShift, DBSCAN, \
    HDBSCAN, SpectralClustering

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from okmich_quant_labelling.utils.label_util import map_label_to_trend_direction
from okmich_quant_ml.hmm import PomegranateHMM, PomegranateMixtureHMM, InferenceMode
from okmich_quant_ml.hmm.pomegranate import DistType
from okmich_quant_features.utils import resample_ohlcv
from okmich_quant_features.utils.transform_pipeline import build_pipeline_from_config
from okmich_quant_ml.regime_filters import BasePostProcessor

#  NOTE: on hmm choice and type of data because Exponential and Gamma distributions only support positive values (x > 0)
#  | Data Type                         | Recommended Distribution      |
#  |-----------------------------------|-------------------------------|
#  | Returns (can be negative)         | Normal, StudentT              |
#  | Volatility (always positive)      | Gamma, Exponential, LogNormal |
#  | Volume (always positive)          | Gamma, LogNormal              |
#  | Scaled features (negative values) | Normal, StudentT              |

# Standard clustering algorithms (non-HMM)
CLUSTERING_ALGOS_STANDARD = [
    "kmeans",
    "mbkmeans",
    "meanshift",
    "dbscan",
    "agglo",
    "ward",
    "birch",
    "hdbscan",
    "spectral",
    "gmm",
]

# HMM-based algorithms
HMM_ALGOS = [
    "hmm_pmgnt",
    "hmm_expnt",
    "hmm_gmma",
    "hmm_lambda",
    "hmm_lognorm",
    "hmm_student",
    "hmm_mm_pmgnt",
    "hmm_mm_expnt",
    "hmm_mm_gmma",
    "hmm_mm_lambda",
    "hmm_mm_lognorm",
    "hmm_mm_student",
]

CLUSTERING_ALGOS = CLUSTERING_ALGOS_STANDARD + HMM_ALGOS


def _train_single_model(name, model, X, sym, label_column_prefix):
    try:
        if name == "gmm":
            labels = model.fit_predict(X)
            silhouette_features = X
        elif name.startswith("hmm"):
            labels = model.fit_predict(X.values).astype(np.int32)
            silhouette_features = X
        else:
            labels = model.fit(X).labels_.astype(np.int32)
            silhouette_features = X

        # Compute silhouette score (only if >1 cluster and no noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        sil_score = None
        if n_clusters > 1:
            if silhouette_features is not None:
                sil_score = silhouette_score(silhouette_features, labels)

        return name, model, labels, sil_score, n_clusters, None  # Return fitted model!
    except Exception as e:
        return name, None, None, None, None, str(e)


def get_naive_signal_generation_fn(**params):
    def naive_signal_generation_fn(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        series = data[params.get("cluster_name", "cluster")]

        long_entries = (series.shift(1) == 1) & (series.shift(2) != 1)
        long_exits = (series.shift(1) != 1) & (series.shift(2) == 1)
        short_entries = (series.shift(1) == -1) & (series.shift(2) != -1)
        short_exits = (series.shift(1) != -1) & (series.shift(2) == -1)

        # Replace NaNs with 0 for clean output
        long_entries = long_entries.fillna(0).to_numpy()
        long_exits = long_exits.fillna(0).to_numpy()
        short_entries = short_entries.fillna(0).to_numpy()
        short_exits = short_exits.fillna(0).to_numpy()

        return long_entries, long_exits, short_entries, short_exits

    return naive_signal_generation_fn


class LabelClusterPipelineConfig:
    def __init__(self, input_dir=".", output_dir=".", should_dim_reduce=True, should_scale=True, should_resample=True,
                 should_save_output_df=True, should_fit_cluster=True, columns_to_scale=None,
                 columns_scaling_exclude: List[str] = None, signal_column=None, timeframe: str = "15min", symbols: List[str] = None,
                 label_column_prefix="lbl_", skip_backtesting: bool = False, mm_n_components: int = 2, data_size: int = -1,
                 training_set_pct: float = 0.75, append_excluded_col_in_result: bool = False, clustering_algos: List[str] = None,
                 offline_labelling_mode: bool = False, inference_mode: InferenceMode = None):
        self.should_dim_reduce = should_dim_reduce
        self.should_scale = should_scale
        self.should_resample = should_resample
        self.should_fit_cluster = should_fit_cluster
        self.save_output_df = should_save_output_df
        self.label_column_prefix = label_column_prefix
        self.columns_to_scale = columns_to_scale if columns_to_scale else []
        self.columns_scaling_exclude = (
            columns_scaling_exclude
            if columns_scaling_exclude
            else ["open", "high", "low", "close", "tick_volume"]
        )
        self.skip_backtesting = skip_backtesting
        self.append_excluded_col_in_result = append_excluded_col_in_result
        self.data_size = data_size
        self.training_set_pct = training_set_pct
        self.signal_column = signal_column if signal_column else []
        self.timeframe = timeframe
        self.symbols = symbols if symbols else []
        self.mm_n_components = mm_n_components
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.clustering_algos = (
            CLUSTERING_ALGOS if clustering_algos is None else clustering_algos
        )
        self.offline_labelling_mode = offline_labelling_mode
        if inference_mode is None:
            self.inference_mode = InferenceMode.SMOOTHING if offline_labelling_mode else InferenceMode.FILTERING
        else:
            self.inference_mode = inference_mode

    def create_features(self, df):
        return df


class LabelAndClusterTestingAndBacktesterPipeline:
    def __init__(self, pipeline_config: LabelClusterPipelineConfig = None, signal_generation_fn: Callable = None,
                 default_cluster: int = 3, transformation_config: dict = None, scaler_type: str = 'standard', **kwargs):
        """
        Parameters
        ----------
        pipeline_config : LabelClusterPipelineConfig
            Pipeline configuration
        signal_generation_fn : Callable, optional
            Custom signal generation function
        default_cluster : int, default 3
            Number of clusters/states
        transformation_config : dict, optional
            Transformation configuration dict. If None, uses empty config (passthrough + scaler only).
        scaler_type : str, default 'standard'
            Type of scaler: 'standard', 'robust', or 'minmax'
        """
        self.pipeline_config = pipeline_config
        self.signal_generation_fn = signal_generation_fn
        self.source_files = (
            pipeline_config.symbols
            if pipeline_config.symbols
            else [
                i.replace(".parquet", "")
                for i in os.listdir(self.pipeline_config.input_dir)
            ]
        )
        self.stats_df = None

        # Build transformation pipeline (empty config = passthrough + scaler)
        self._transform_pipeline = build_pipeline_from_config(
            transformation_config or {},
            scaler_type=scaler_type
        )

        self.pca = None  # PCA handled separately for adaptive fitting
        self.cluster_columns = None
        self.random_seed = 1
        self.algo_silhouette_scores = {}
        self.cluster_algorithms = {}
        inference_mode = self.pipeline_config.inference_mode
        if not self.pipeline_config.offline_labelling_mode:
            if not (0.0 < self.pipeline_config.training_set_pct < 1.0):
                raise ValueError(
                    f"The training set pct value must be between 0 and 1, got {self.pipeline_config.training_set_pct}"
                )

        if "kmeans" in self.pipeline_config.clustering_algos:
            self.cluster_algorithms["kmeans"] = KMeans(
                n_clusters=default_cluster, random_state=self.random_seed
            )
        if "mbkmeans" in self.pipeline_config.clustering_algos:
            self.cluster_algorithms["mbkmeans"] = MiniBatchKMeans(
                n_clusters=default_cluster,
                batch_size=20000,
                random_state=self.random_seed,
            )
        if "meanshift" in self.pipeline_config.clustering_algos:
            self.cluster_algorithms["meanshift"] = MeanShift(
                bandwidth=None, n_jobs=-1
            )  # bloated regime count > 100
        if "dbscan" in self.pipeline_config.clustering_algos:
            self.cluster_algorithms["dbscan"] = DBSCAN(
                eps=0.5, min_samples=default_cluster, n_jobs=-1
            )  # bloated regime count > 100
        if "agglo" in self.pipeline_config.clustering_algos:
            self.cluster_algorithms["agglo"] = AgglomerativeClustering(
                n_clusters=default_cluster,
                linkage="average",
            )
        if "ward" in self.pipeline_config.clustering_algos:
            self.cluster_algorithms["ward"] = AgglomerativeClustering(
                n_clusters=default_cluster, linkage="ward"
            )
        if "birch" in self.pipeline_config.clustering_algos:
            self.cluster_algorithms["birch"] = Birch(n_clusters=default_cluster)
        if "gmm" in self.pipeline_config.clustering_algos:
            self.cluster_algorithms["gmm"] = GaussianMixture(
                n_components=default_cluster, random_state=self.random_seed
            )
        if "hdbscan" in self.pipeline_config.clustering_algos:
            self.cluster_algorithms["hdbscan"] = HDBSCAN(
                min_cluster_size=100,
                min_samples=10,
                cluster_selection_epsilon=0.0,
                metric="euclidean",
                algorithm="ball_tree",
                n_jobs=-1,
            )
        if "spectral" in self.pipeline_config.clustering_algos:
            self.cluster_algorithms["spectral"] = SpectralClustering(
                n_clusters=default_cluster,
                assign_labels="kmeans",
                affinity="precomputed",
                eigen_solver="arpack",
                n_jobs=-1,
            )
        hmm_single = {
            "hmm_pmgnt": DistType.NORMAL, "hmm_expnt": DistType.EXPONENTIAL,
            "hmm_gmma": DistType.GAMMA, "hmm_lambda": DistType.LAMDA,
            "hmm_lognorm": DistType.LOGNORMAL, "hmm_student": DistType.STUDENTT,
        }
        for algo_key, dist_type in hmm_single.items():
            if algo_key in self.pipeline_config.clustering_algos:
                self.cluster_algorithms[algo_key] = PomegranateHMM(
                    distribution_type=dist_type, n_states=default_cluster,
                    inference_mode=inference_mode, random_state=self.random_seed,
                )
        hmm_mixture = {
            "hmm_mm_pmgnt": DistType.NORMAL, "hmm_mm_expnt": DistType.EXPONENTIAL,
            "hmm_mm_gmma": DistType.GAMMA, "hmm_mm_lambda": DistType.LAMDA,
            "hmm_mm_lognorm": DistType.LOGNORMAL, "hmm_mm_student": DistType.STUDENTT,
        }
        for algo_key, dist_type in hmm_mixture.items():
            if algo_key in self.pipeline_config.clustering_algos:
                self.cluster_algorithms[algo_key] = PomegranateMixtureHMM(
                    distribution_type=dist_type, n_states=default_cluster, inference_mode=inference_mode,
                    n_components=self.pipeline_config.mm_n_components, random_state=self.random_seed,
                )
    def get_supported_clustering_algos(self):
        return CLUSTERING_ALGOS

    def get_data(self, symbol: str) -> pd.DataFrame:
        df = pd.read_parquet(
            os.path.join(self.pipeline_config.input_dir, f"{symbol}.parquet")
        )
        if "time" in df.columns:
            df.set_index("time", inplace=True)
        if "spread" in df.columns:
            df.drop(columns=["spread"], axis=1, inplace=True)
        if "real_volume" in df.columns:
            df.drop(columns=["real_volume"], axis=1, inplace=True)
        return df

    def resample(self, df):
        return resample_ohlcv(df, self.pipeline_config.timeframe).dropna()

    def feature_engineering(self, df):
        feat_df = self.pipeline_config.create_features(df)
        feat_cold = [
            c
            for c in feat_df.columns
            if c not in self.pipeline_config.columns_scaling_exclude
        ]
        return feat_df[feat_cold]

    def scale_features(self, df):
        columns_to_scale = [
            c
            for c in (self.pipeline_config.columns_to_scale or df.columns)
            if c != "close"
        ]
        df[columns_to_scale] = self._transform_pipeline.fit_transform(df[columns_to_scale])
        # Select features: scaled + binary flags
        feature_cols = columns_to_scale + [
            col
            for col in df.columns
            if col not in columns_to_scale and df[col].isin([-1, 0, 1]).all()
        ]
        print(feature_cols)
        return df[feature_cols].copy()

    def reduce_dimensionality(self, df):
        n_dimensions = min(5, df.shape[1])
        while True:
            pca = PCA(n_components=n_dimensions, random_state=self.random_seed)
            df_pca = pd.DataFrame(pca.fit_transform(df))
            variance_explained = np.sum(pca.explained_variance_ratio_)
            if variance_explained >= 0.95:
                print(
                    f"\t => Dimension reduced from {df.shape[1]} to {n_dimensions}. Variance explained: {variance_explained}"
                )
                break

            n_dimensions += 1
        # Store the fitted PCA transformer for later use
        self.pca = pca
        return df_pca

    def fit_and_predict_cluster(self, sym, X, df, n_jobs=-1, backend="loky"):
        n_models = len(self.cluster_algorithms)
        cores_used = n_jobs if n_jobs > 0 else "all available"
        print(
            f"\t => 🚀 Starting parallel training of {n_models} models using {cores_used} cores (backend: {backend})..."
        )

        # Train all models in parallel
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=0, return_as="generator")(
            delayed(_train_single_model)(
                name, model, X, sym, self.pipeline_config.label_column_prefix
            )
            for name, model in self.cluster_algorithms.items()
        )

        # Process results and update dataframe
        successful_models = 0
        failed_models = []
        for name, fitted_model, labels, sil_score, n_clusters, error in results:
            if error is None and labels is not None:
                # Store the fitted model back into cluster_algorithms
                self.cluster_algorithms[name] = fitted_model

                df[f"{self.pipeline_config.label_column_prefix}{name}"] = labels
                if sil_score is not None:
                    self.algo_silhouette_scores[name] = sil_score
                    print(
                        f"\t\t => 🔍 Trained {name.upper()} model ....... \t\t✅ Saved: Clusters: {n_clusters} | Silhouette: {round(sil_score, 4)}"
                    )
                else:
                    print(
                        f"\t\t => 🔍 Trained {name.upper()} model ....... \t\t✅ Saved: Clusters: {n_clusters} | ❌ But Silhouette is None"
                    )
                successful_models += 1
            else:
                print(
                    f"\t\t => 🔍 Trained {name.upper()} model ....... \t\t❌ {name} failed on {sym}: {error}"
                )
                failed_models.append(name)

        print(
            f"\n\t => ✨ Completed: {successful_models}/{n_models} models trained successfully"
        )
        return df

    def predict_cluster(self, sym, X, df, model_key):
        model = self.cluster_algorithms[model_key]
        probs = None
        try:
            print(
                f"\t => 🔍 Predicting with {model_key.upper()} model ....... ", end=" "
            )
            if model_key == "gmm":
                labels = model.predict(X)
                probs = model.predict_proba(X)
                silhouette_features = X
            elif model_key.startswith("hmm"):
                labels = model.predict(X.values).astype(np.int32)
                if model.inference_mode != InferenceMode.VITERBI:
                    probs = model.predict_proba(X.values)
                silhouette_features = X
            else:
                labels = model.predict(X).astype(np.int32)
                silhouette_features = X

            df[f"{self.pipeline_config.label_column_prefix}{model_key}"] = labels

            # Compute silhouette score (only if >1 cluster and no noise)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            sil_score = None
            if n_clusters > 1 and len(labels) == len(df):
                if silhouette_features is not None and silhouette_features.shape[
                    0
                ] == len(labels):
                    sil_score = silhouette_score(silhouette_features, labels)

            print(f"\t✅ Done: Clusters: {n_clusters} | Silhouette: {sil_score}")
        except Exception as e:
            print(f"\t❌ {model_key} failed on {sym}: {e}")
        return df, probs

    def map_clusters_to_signal(self, df):
        self.cluster_columns = [
            i
            for i in df.columns
            if i.startswith(self.pipeline_config.label_column_prefix)
        ]
        returns_col = "returns_1"
        if returns_col not in df.columns:
            df[returns_col] = np.log(df["close"] / df["close"].shift(1))

        for col in self.cluster_columns:
            cluster_to_signal = map_label_to_trend_direction(
                df, state_col=col, return_col=returns_col
            )
            df[col] = df[col].map(cluster_to_signal)
        return df

    def run_backtest(self, df):
        self.stats_df = None
        for col_name in self.cluster_columns:
            signal_generator_fn = (
                self.signal_generation_fn
                or get_naive_signal_generation_fn(**{"cluster_name": col_name})
            )
            # Generate signals
            long_entries, long_exits, short_entries, short_exits = signal_generator_fn(df)

            # Constants
            initial_capital = 10000  # USD
            contract_size = 1
            pip_value = 1.0  # e.g., 1 pip = 0.10 USD for 0.01 lot on
            position_size = 0.1
            # Use vectorbt's built-in conflict resolution
            portfolio = vbt.Portfolio.from_signals(
                init_cash=initial_capital,
                fees=0.00,
                size=position_size,
                close=df["close"],
                entries=long_entries,
                exits=long_exits,
                short_entries=short_entries,
                short_exits=short_exits,
                upon_dir_conflict=vbt.portfolio.enums.DirectionConflictMode.Opposite,
                upon_opposite_entry="close",
                freq=self.pipeline_config.timeframe,
            )
            stats = portfolio.stats()
            stats = stats.to_frame()
            stats.columns = [col_name]
            if self.stats_df is None:
                self.stats_df = stats
            else:
                self.stats_df = pd.concat([self.stats_df, stats], axis=1)

    def run(self):
        def do_feature_engineering(df):
            raw_features = self.feature_engineering(df)
            features = (
                self.scale_features(raw_features)
                if self.pipeline_config.should_scale
                else raw_features
            )
            scaled_features = (
                self.reduce_dimensionality(features)
                if self.pipeline_config.should_dim_reduce
                else features
            )
            return scaled_features, raw_features

        output_dir = self.pipeline_config.output_dir
        output_df = None
        for sym in self.source_files:
            print(f">>>>>>>> {sym}")
            df = self.get_data(sym)
            df = (
                df
                if self.pipeline_config.data_size == -1
                else df[-self.pipeline_config.data_size :]
            )
            df = self.resample(df) if self.pipeline_config.should_resample else df

            # Check if we're in offline labelling mode
            if self.pipeline_config.offline_labelling_mode:
                print(
                    f"\t => Offline labelling mode: Using entire dataset ({len(df)} rows) for fitting and labelling"
                )
                # Use entire dataset for feature engineering, fitting and prediction
                features, df_features = do_feature_engineering(df)
                output_df = (
                    self.fit_and_predict_cluster(sym, features, df_features)
                    if self.pipeline_config.should_fit_cluster
                    else df_features
                )
            else:
                # Original train/test split mode
                # Split data into train and test sets based on training_set_pct
                train_size = int(len(df) * self.pipeline_config.training_set_pct)
                df_train = df.iloc[:train_size, :]
                df_test = df.iloc[train_size:, :]
                print(
                    f"\t => Split data: Train={len(df_train)} rows ({self.pipeline_config.training_set_pct*100:.0f}%), Test={len(df_test)} rows ({(1-self.pipeline_config.training_set_pct)*100:.0f}%)"
                )

                # Feature engineering on training data
                train_features, df_features_train = do_feature_engineering(df_train)

                # Fit models on training set
                if self.pipeline_config.should_fit_cluster:
                    print(f"\t => Training models on training set...")
                    _ = self.fit_and_predict_cluster(sym, train_features, df_features_train)

                    # Now run inference on test set
                    print(f"\t => Running inference on test set...")
                    df_features_test = self.feature_engineering(df_test)

                    # Transform test features using fitted pipeline/PCA (don't fit again!)
                    if self.pipeline_config.should_scale:
                        columns_to_scale = [
                            c
                            for c in (
                                self.pipeline_config.columns_to_scale
                                or df_features_test.columns
                            )
                            if c != "close"
                        ]
                        df_features_test[columns_to_scale] = self._transform_pipeline.transform(
                            df_features_test[columns_to_scale]
                        )
                        feature_cols = columns_to_scale + [
                            col
                            for col in df_features_test.columns
                            if col not in columns_to_scale
                            and df_features_test[col].isin([-1, 0, 1]).all()
                        ]
                        test_features = df_features_test[feature_cols].copy()
                    else:
                        test_features = df_features_test

                    if self.pipeline_config.should_dim_reduce and self.pca is not None:
                        test_features = pd.DataFrame(self.pca.transform(test_features))

                    # Run inference on test set for all trained models
                    output_df = df_features_test.copy()
                    for model_key in self.cluster_algorithms.keys():
                        output_df, _ = self.predict_cluster(
                            sym, test_features, output_df, model_key
                        )
                else:
                    output_df = self.feature_engineering(df_test)

            # Append excluded columns from the entire dataset
            if self.pipeline_config.append_excluded_col_in_result:
                output_df = df[self.pipeline_config.columns_scaling_exclude].join(
                    output_df
                )

            # Save output (entire dataset)
            if self.pipeline_config.save_output_df:
                output_df.to_parquet(os.path.join(output_dir, f"{sym}.parquet"))

            # Run backtesting on entire dataset
            if not self.pipeline_config.skip_backtesting:
                signal_mapped_df = self.map_clusters_to_signal(output_df)
                self.run_backtest(signal_mapped_df)
                # create the output dir if it does not exist
                os.makedirs(self.pipeline_config.output_dir, exist_ok=True)
                self.stats_df.to_excel(
                    os.path.join(self.pipeline_config.output_dir, f"{sym}.xlsx")
                )

        return output_df if len(self.source_files) == 1 else None

    def get_transform_pipeline(self):
        """
        Get the full transformation pipeline including PCA if fitted.
        """
        if self.pca is None:
            return self._transform_pipeline
        else:
            from sklearn.pipeline import Pipeline

            # Get existing steps from _transform_pipeline
            existing_steps = list(self._transform_pipeline.steps)
            existing_steps.append(('pca', self.pca))
            # Return new pipeline with all steps
            return Pipeline(existing_steps)

    def get_model(self, model_key):
        if model_key in self.cluster_algorithms:
            return self.cluster_algorithms[model_key]
        else:
            raise KeyError(f"{model_key} is not a valid model key")

    def run_inference(self, model_key):
        if self.pipeline_config.offline_labelling_mode:
            raise ValueError("No inference running when offline_labelling_mode is True")

        if model_key not in self.cluster_algorithms:
            raise KeyError(f"{model_key} is not a valid model key")

        output_dfs = {}
        for sym in self.source_files:
            print(f">>>>>>>> {sym}")
            df = self.get_data(sym)
            df = df if self.pipeline_config.data_size == -1 else df[-self.pipeline_config.data_size :]
            df = self.resample(df) if self.pipeline_config.should_resample else df

            # Split data into train and test sets - use test set for inference
            train_size = int(len(df) * self.pipeline_config.training_set_pct)
            df_test = df.iloc[train_size:, :]
            print(
                f"\t => Using test set for inference: {len(df_test)} rows ({(1-self.pipeline_config.training_set_pct)*100:.0f}%)"
            )

            # Feature engineering and transformation on test data
            df_features_test = self.feature_engineering(df_test)

            # Transform test features using fitted pipeline/PCA (don't fit again!)
            if self.pipeline_config.should_scale:
                if self._transform_pipeline is None:
                    raise ValueError(
                        "Transform pipeline is not fitted. Please run training first using run() method."
                    )
                columns_to_scale = [
                    c for c in (self.pipeline_config.columns_to_scale or df_features_test.columns) if c != "close"
                ]
                df_features_test[columns_to_scale] = self._transform_pipeline.transform(
                    df_features_test[columns_to_scale]
                )
                feature_cols = columns_to_scale + [
                    col for col in df_features_test.columns if col not in columns_to_scale
                    and df_features_test[col].isin([-1, 0, 1]).all()
                ]
                test_features = df_features_test[feature_cols].copy()
            else:
                test_features = df_features_test

            if self.pipeline_config.should_dim_reduce:
                if self.pca is None:
                    raise ValueError(
                        "PCA is not fitted. Please run training first using run() method."
                    )
                test_features = pd.DataFrame(self.pca.transform(test_features))

            # Run inference on test set
            output_df, prob_matrix = self.predict_cluster(
                sym, test_features, df_features_test, model_key
            )

            # Append excluded columns from test set
            if self.pipeline_config.append_excluded_col_in_result:
                output_df = df_test[self.pipeline_config.columns_scaling_exclude].join(
                    output_df
                )

            output_dfs[sym] = (output_df, prob_matrix)
        return output_dfs

    def post_processing(self, output_df, post_processor: BasePostProcessor):
        result_df = output_df.copy()
        columns = [
            c
            for c in result_df.columns
            if c.startswith(self.pipeline_config.label_column_prefix)
        ]
        for col in columns:
            result_df[col] = post_processor.process(result_df[col])
        return result_df
