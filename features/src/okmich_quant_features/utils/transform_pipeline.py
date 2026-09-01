import json
import joblib
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer

from .transform import (YeoJohnsonTransformer, BoxCoxTransformer, LogitTransformer, LogTransformer,
                        QUANTILE_OR_RANK)

# Transformation types the config understands, beyond 'passthrough'. Scanned in the order the
# recommender wrote them, so a row's own priority wins rather than a fixed global one.
CONFIG_TRANSFORMATIONS = ('yeo-johnson', 'box-cox', 'logit', 'log', 'standardize')


def config_transformation_type(transformation_str) -> str:
    """
    Map a recommendation string to a config transformation type.

    The previous implementation checked ``'standardize' in trans_str`` first and returned
    'passthrough' on a hit, so a compound recommendation such as ``'box-cox, standardize'`` --
    which FeatureEDA emits routinely -- silently discarded the box-cox and exported the feature
    as untransformed. Scanning tokens in order fixes that and additionally lets 'standardize'
    and 'quantile/rank' reach the config as themselves instead of being dropped.

    Parameters
    ----------
    transformation_str : str or NaN
        Comma-separated recommendation, e.g. 'logit, standardize'.

    Returns
    -------
    str
        A member of CONFIG_TRANSFORMATIONS, 'quantile', or 'passthrough'.
    """
    if transformation_str is None:
        return 'passthrough'
    if not isinstance(transformation_str, str):
        try:
            if pd.isna(transformation_str):
                return 'passthrough'
        except (TypeError, ValueError):
            pass
        transformation_str = str(transformation_str)

    for token in transformation_str.split(','):
        token = token.strip()
        if token == QUANTILE_OR_RANK:
            return 'quantile'
        if token in CONFIG_TRANSFORMATIONS:
            return token
    return 'passthrough'


def encode_transformation_recommendation(transformation_df, description: str = None) -> Dict:
    config = {
        'description': description or 'Feature transformation configuration',
        'transformations': {}
    }

    for idx, row in transformation_df.iterrows():
        feature = row['feature']
        config['transformations'][feature] = {
            'type': config_transformation_type(row['transformations']),
            'reason': row.get('reason', ''),
        }

    return config


def export_transformation_config(transformation_df, output_path: str, description: str = None) -> Dict:
    # Generate config
    config = encode_transformation_recommendation(transformation_df, description)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[ok] Exported config to {output_path}")
    print(f"  {len(config['transformations'])} features")
    print(f"  Edit this file to customize before training")

    return config


def load_transformation_config(config_path: str) -> Dict:
    """Load transformation config from JSON."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"[ok] Loaded config: {len(config['transformations'])} features")
    return config


def build_pipeline_from_config(config: Union[str, Dict], scaler_type: str = 'robust', fitted_scaler=None,
                               post_transformers: List[tuple] = None) -> Pipeline:
    # Load config if path
    if isinstance(config, str):
        config = load_transformation_config(config)

    # Group features by transformation type (exclude 'passthrough')
    groups = {
        'yeo-johnson': [],
        'box-cox': [],
        'logit': [],
        'log': [],
        'quantile': [],
        'standardize': [],
    }

    if 'transformations' in config:
        for feature, trans_config in config['transformations'].items():
            trans_type = trans_config.get('type', 'passthrough')
            if trans_type in groups:
                groups[trans_type].append(feature)

    # Build transformers list (only for features needing transformation)
    transformers = []

    if groups['yeo-johnson']:
        transformers.append(('yeo_johnson', YeoJohnsonTransformer(standardize=False), groups['yeo-johnson']))

    if groups['box-cox']:
        transformers.append(('box_cox', BoxCoxTransformer(standardize=False), groups['box-cox']))

    if groups['logit']:
        transformers.append(('logit', LogitTransformer(), groups['logit']))

    if groups['log']:
        transformers.append(('log', LogTransformer(), groups['log']))

    # A quantile transform is not affine, so no scaler downstream can stand in for it. Heavy-
    # tailed features previously encoded as 'passthrough' therefore reached the model untreated.
    if groups['quantile']:
        transformers.append(('quantile', QuantileTransformer(output_distribution='normal'), groups['quantile']))

    # Standardisation IS affine, so a global scaler step already covers it. Adding a per-column
    # StandardScaler underneath the default RobustScaler would just scale these columns twice,
    # so the step is only materialised when no global scaler is configured. The config still
    # records 'standardize' faithfully either way.
    standardize_subsumed = bool(groups['standardize']) and (fitted_scaler is not None or scaler_type is not None)
    if groups['standardize'] and not standardize_subsumed:
        transformers.append(('standardize', StandardScaler(), groups['standardize']))

    # Create ColumnTransformer - passthrough columns not in config
    col_transformer = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough',  # No columns dropped
        verbose_feature_names_out=False,
    )

    # Build pipeline steps
    steps = [('transforms', col_transformer)]

    # Add scaler if specified
    if fitted_scaler is not None:
        steps.append(('scaler', fitted_scaler)) # Use pre-fitted scaler (ignores scaler_type)
    elif scaler_type == 'robust':
        steps.append(('scaler', RobustScaler()))
    elif scaler_type == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaler_type == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    elif scaler_type is not None:
        raise ValueError(f"Unknown scaler: {scaler_type}. Use 'robust', 'standard', 'minmax', or None")

    # Add post-scaling transformers (PCA, feature selection, etc.)
    if post_transformers:
        for name, transformer in post_transformers:
            steps.append((name, transformer))

    pipeline = Pipeline(steps)

    # Print summary
    n_transformers = len(transformers)
    scaler_desc = 'fitted scaler' if fitted_scaler is not None else scaler_type
    has_scaler = fitted_scaler is not None or scaler_type is not None
    n_post = len(post_transformers) if post_transformers else 0

    if n_transformers == 0 and not has_scaler and n_post == 0:
        print(f"[ok] Built passthrough pipeline (identity transform)")
    elif n_transformers == 0 and n_post == 0:
        print(f"[ok] Built pipeline: Passthrough -> {scaler_desc}")
    else:
        print(f"[ok] Built pipeline:")
        print(f"  Transformations: {n_transformers} groups + passthrough others")
        print(f"  Scaler: {scaler_desc or 'None'}")
        if standardize_subsumed:
            print(f"  Note: {len(groups['standardize'])} 'standardize' feature(s) left to the "
                  f"{scaler_desc} step rather than scaled twice")
        if n_post > 0:
            post_names = [name for name, _ in post_transformers]
            print(f"  Post-transformers: {', '.join(post_names)}")

    return pipeline


def save_pipeline_artifacts(pipeline: Pipeline, feature_cols: List[str], output_dir: str, metadata: Dict = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, output_dir / 'pipeline.pkl')
    # Save feature list
    with open(output_dir / 'feature_cols.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    # Save metadata
    if metadata:
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    print(f"\n[ok] Saved artifacts to {output_dir}/")
    print(f"  - pipeline.pkl")
    print(f"  - feature_cols.json")
    if metadata:
        print(f"  - metadata.json")


def load_pipeline_artifacts(artifacts_dir: str) -> Dict:
    artifacts_dir = Path(artifacts_dir)
    # Load pipeline
    pipeline = joblib.load(artifacts_dir / 'pipeline.pkl')

    # Load feature cols
    with open(artifacts_dir / 'feature_cols.json', 'r') as f:
        feature_cols = json.load(f)

    # Load metadata if exists
    metadata = None
    metadata_path = artifacts_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    print(f"[ok] Loaded artifacts from {artifacts_dir}/")
    print(f"  {len(feature_cols)} features")
    if metadata:
        print(f"  Version: {metadata.get('version', 'N/A')}")

    return {
        'pipeline': pipeline,
        'feature_cols': feature_cols,
        'metadata': metadata,
    }
