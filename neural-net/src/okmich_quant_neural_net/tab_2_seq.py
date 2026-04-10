import warnings
from copy import deepcopy
from typing import Tuple, Optional

import numpy as np


def transform_to_sequence(features: np.ndarray, sequence_length: int, scaler: object) -> np.ndarray:
    """
    Create sequences from raw data using a fitted scaler for inference

    This function is designed for production use where you have:
    - New/unseen data that needs to be transformed into sequences
    - An already-fitted scaler from training

    Parameters:
    -----------
    features : np.ndarray
        Test features (2D array: samples × features)
    sequence_length : int
        Number of time steps in each sequence
    scaler : sklearn scaler object

    Returns:
    --------
    features_seq : np.ndarray
        Test sequences, shape (n_samples - sequence_length, sequence_length, n_features)
    """
    if len(features.shape) != 2:
        raise ValueError("features must be 2-dimensional (samples, features)")

    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    if len(features) <= sequence_length:
        raise ValueError(
            f"features has {len(features)} samples but sequence_length is {sequence_length}. "
            f"Need at least {sequence_length + 1} samples."
        )

    features_scaled = scaler.transform(features)

    # Create sequences using vectorized numpy operations
    n_samples = len(features_scaled) - sequence_length
    # n_features = features_scaled.shape[1]

    # Use advanced indexing to create all sequences at once
    indices = np.arange(sequence_length)[None, :] + np.arange(n_samples)[:, None]
    features_seq = features_scaled[indices]

    return features_seq


def create_sequences(features: np.ndarray, target: np.ndarray, sequence_length: int, scaler: object,
                     fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from test data using a fitted or unfitted scaler.

    Parameters:
    -----------
    features : np.ndarray
        Test features (2D array: samples × features)
    target : np.ndarray
        Test targets (1D array)
    sequence_length : int
        Number of time steps in each sequence
    scaler : sklearn scaler object
    fit_scaler : bool
        fit_transform using the scaler if True, otherwise, transform data using the scaler

    Returns:
    --------
    features_seq : np.ndarray
        Test sequences, shape (n_samples - sequence_length, sequence_length, n_features)
    target_seq : np.ndarray
        Test targets aligned with sequences, shape (n_samples - sequence_length,)
    """
    if len(features.shape) != 2:
        raise ValueError("features must be 2-dimensional (samples, features)")

    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    if len(features) <= sequence_length:
        raise ValueError(
            f"features has {len(features)} samples but sequence_length is {sequence_length}. "
            f"Need at least {sequence_length + 1} samples."
        )

    if len(features) != len(target):
        raise ValueError(f"features ({len(features)}) and target ({len(target)}) must have same length")

    # Scale features using fitted scaler
    if fit_scaler:
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)

    # Create sequences using vectorized numpy operations
    n_samples = len(features_scaled) - sequence_length

    # Create feature sequences using advanced indexing
    indices = np.arange(sequence_length)[None, :] + np.arange(n_samples)[:, None]
    features_seq = features_scaled[indices]

    # Create target sequences - targets are at position sequence_length after each window starts
    target_seq = target[sequence_length:]

    return features_seq, target_seq


def create_train_test_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int, test_size: float = 0.2,
                                shuffle: bool = False, random_state: int = None, scaler: Optional[object] = None,
                                scale_target: bool = False,
                                split_mode: str = 'continuation') -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[object], Optional[object]]:
    """
    Creates 3D sequence data suitable for RNN-type models from tabular data with optional scaling.

    Parameters:
    -----------
    X : np.ndarray
        Input features (2D array: samples × features)
    y : np.ndarray
        Target variable (1D array)
    sequence_length : int
        Number of time steps in each sequence
    test_size : float, default=0.2
        Proportion of data to use for testing (0.0 to 1.0)
    shuffle : bool, default=False
        Whether to shuffle the training sequences.  Set True only for
        stateless models (e.g., Transformers).  For stateful / recurrent
        models, shuffling breaks temporal order and corrupts state propagation.
        Test sequences are never shuffled regardless of this flag.
    random_state : int, optional
        Random seed for reproducibility when shuffle=True
    scaler : sklearn scaler object, optional
        Scaler for features (e.g., StandardScaler()). If None, no scaling applied.
    scale_target : bool, default=False
        Whether to scale the target variable
    split_mode : {'continuation', 'strict_disjoint'}, default='continuation'
        Controls how raw data is divided before sequence creation.

        ``'continuation'`` (default):
            The first test sequence begins ``sequence_length`` rows before the
            nominal split point, so it can look back into the training period.
            This creates an intentional raw-data overlap of ``sequence_length``
            rows between train and test slices — the overlap is used only as
            lookback context, never as a prediction target.  Correct for
            stateful / recurrent models where continuity matters.

        ``'strict_disjoint'``:
            Train and test raw slices do not share any rows.  The first test
            sequence starts exactly at the split point with no lookback into
            the training region.  Use this when evaluating on completely
            held-out data or when the evaluation framework requires index
            disjointness.

    Returns:
    --------
    X_train : np.ndarray
        Training sequences, shape (n_train_samples, sequence_length, n_features)
    y_train : np.ndarray
        Training targets, shape (n_train_samples,)
    X_test : np.ndarray
        Test sequences, shape (n_test_samples, sequence_length, n_features)
    y_test : np.ndarray
        Test targets, shape (n_test_samples,)
    feature_scaler : object or None
        Fitted feature scaler (for inverse transform)
    target_scaler : object or None
        Fitted target scaler (for inverse transform)
    """
    # Convert to numpy arrays if needed
    X = np.array(X)
    y = np.array(y)

    # Validate input
    if len(X.shape) != 2:
        raise ValueError("X must be 2-dimensional")

    if len(y.shape) != 1:
        raise ValueError("y must be 1-dimensional")

    if len(X) != len(y):
        raise ValueError(f"X ({len(X)}) and y ({len(y)}) must have same length")

    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    if sequence_length >= len(X):
        raise ValueError("sequence_length must be less than the number of samples")

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    if split_mode not in ('continuation', 'strict_disjoint'):
        raise ValueError(f"split_mode must be 'continuation' or 'strict_disjoint', got '{split_mode}'")

    if shuffle:
        warnings.warn("shuffle=True randomises the order of training sequences. "
                      "For stateful / recurrent models this breaks temporal order and corrupts "
                      "hidden-state propagation. Only use shuffle=True with stateless models "
                      "(e.g. Transformers, standard classifiers).", UserWarning, stacklevel=2)

    # Determine split point BEFORE creating sequences — scaler is fit only on training rows.
    n_sequences = len(X) - sequence_length
    split_idx = int(n_sequences * (1 - test_size))

    if split_mode == 'continuation':
        # Train slice ends sequence_length rows past the split so the first test
        # sequence can look back into the training region (intentional overlap).
        train_end_idx = split_idx + sequence_length
        test_start_idx = split_idx
    else:  # strict_disjoint
        # No raw-data overlap between train and test slices.
        train_end_idx = split_idx
        test_start_idx = split_idx

    # Split the raw data first
    train_features = X[:train_end_idx]
    test_features = X[test_start_idx:]
    train_targets = y[:train_end_idx]
    test_targets = y[test_start_idx:]

    # Apply scaling if requested
    feature_scaler = None
    target_scaler = None

    if scaler is not None:
        feature_scaler = deepcopy(scaler)
        # Fit on train, transform both
        train_features = feature_scaler.fit_transform(train_features)
        test_features = feature_scaler.transform(test_features)

    if scale_target and scaler is not None:
        target_scaler = deepcopy(scaler)
        # Fit on train, transform both
        train_targets = target_scaler.fit_transform(
            train_targets.reshape(-1, 1)
        ).ravel()
        test_targets = target_scaler.transform(test_targets.reshape(-1, 1)).ravel()

    # Create sequences from scaled data using vectorized numpy operations
    n_train_samples = len(train_features) - sequence_length
    n_test_samples = len(test_features) - sequence_length

    # Create training sequences using advanced indexing
    train_indices = np.arange(sequence_length)[None, :] + np.arange(n_train_samples)[:, None]
    X_train = train_features[train_indices]
    y_train = train_targets[sequence_length:]

    # Create test sequences using advanced indexing
    test_indices = np.arange(sequence_length)[None, :] + np.arange(n_test_samples)[:, None]
    X_test = test_features[test_indices]
    y_test = test_targets[sequence_length:]

    # Shuffle training sequences if requested (after scaling and sequence creation).
    # Test sequences are never shuffled — temporal order in evaluation must be preserved.
    if shuffle:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler


def create_multi_output_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int, test_size: float = 0.2,
                                  shuffle: bool = False, random_state: int = None, scaler: Optional[object] = None,
                                  scale_target: bool = False, split_mode: str = 'continuation'
                                  ) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[object], Optional[object]]:
    """
    Creates 3D sequence data suitable for multi-output RNN models with optional scaling.

    Parameters:
    -----------
    X : np.ndarray
        Input features (2D array: samples × features)
    y : np.ndarray
        Target variables (2D array: samples × targets)
    sequence_length : int
        Number of time steps in each sequence
    test_size : float, default=0.2
        Proportion of data to use for testing (0.0 to 1.0)
    shuffle : bool, default=False
        Whether to shuffle the training sequences.  Set True only for
        stateless models.  Test sequences are never shuffled.
    random_state : int, optional
        Random seed for reproducibility when shuffle=True
    scaler : sklearn scaler object, optional
        Scaler for features (e.g., StandardScaler()). If None, no scaling applied.
    scale_target : bool, default=False
        Whether to scale the target variables
    split_mode : {'continuation', 'strict_disjoint'}, default='continuation'
        See ``create_train_test_sequences`` for full documentation of each mode.
        ``'continuation'``: intentional raw-data overlap of ``sequence_length``
        rows so the first test sequence can look back into the training region.
        ``'strict_disjoint'``: no shared raw rows between train and test.

    Returns:
    --------
    X_train : np.ndarray
        Training sequences, shape (n_train_samples, sequence_length, n_features)
    y_train : np.ndarray
        Training targets, shape (n_train_samples, n_targets)
    X_test : np.ndarray
        Test sequences, shape (n_test_samples, sequence_length, n_features)
    y_test : np.ndarray
        Test targets, shape (n_test_samples, n_targets)
    feature_scaler : object or None
        Fitted feature scaler (for inverse transform)
    target_scaler : object or None
        Fitted target scaler (for inverse transform)
    """
    # Convert to numpy arrays if needed
    X = np.array(X)
    y = np.array(y)

    # Validate input
    if len(X.shape) != 2:
        raise ValueError("X must be 2-dimensional")

    if len(y.shape) != 2:
        raise ValueError("y must be 2-dimensional for multi-output")

    if len(X) != len(y):
        raise ValueError(f"X ({len(X)}) and y ({len(y)}) must have same length")

    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    if sequence_length >= len(X):
        raise ValueError("sequence_length must be less than the number of samples")

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    if split_mode not in ('continuation', 'strict_disjoint'):
        raise ValueError(f"split_mode must be 'continuation' or 'strict_disjoint', got '{split_mode}'")

    if shuffle:
        warnings.warn("shuffle=True randomises the order of training sequences. "
                      "For stateful / recurrent models this breaks temporal order and corrupts "
                      "hidden-state propagation. Only use shuffle=True with stateless models.",
                      UserWarning, stacklevel=2)

    # Determine split point BEFORE creating sequences
    n_sequences = len(X) - sequence_length
    split_idx = int(n_sequences * (1 - test_size))

    if split_mode == 'continuation':
        train_end_idx = split_idx + sequence_length
        test_start_idx = split_idx
    else:  # strict_disjoint
        train_end_idx = split_idx
        test_start_idx = split_idx

    # Split the raw data first
    train_features = X[:train_end_idx]
    test_features = X[test_start_idx:]
    train_targets = y[:train_end_idx]
    test_targets = y[test_start_idx:]

    # Apply scaling if requested
    feature_scaler = None
    target_scaler = None

    if scaler is not None:
        feature_scaler = deepcopy(scaler)
        # Fit on train, transform both
        train_features = feature_scaler.fit_transform(train_features)
        test_features = feature_scaler.transform(test_features)

    if scale_target and scaler is not None:
        target_scaler = deepcopy(scaler)

        # Fit on train, transform both
        train_targets = target_scaler.fit_transform(train_targets)
        test_targets = target_scaler.transform(test_targets)

    # Create sequences from scaled data using vectorized numpy operations
    n_train_samples = len(train_features) - sequence_length
    n_test_samples = len(test_features) - sequence_length

    # Create training sequences using advanced indexing
    train_indices = np.arange(sequence_length)[None, :] + np.arange(n_train_samples)[:, None]
    X_train = train_features[train_indices]
    y_train = train_targets[sequence_length:]

    # Create test sequences using advanced indexing
    test_indices = np.arange(sequence_length)[None, :] + np.arange(n_test_samples)[:, None]
    X_test = test_features[test_indices]
    y_test = test_targets[sequence_length:]

    # Shuffle training sequences only; test sequences keep temporal order.
    if shuffle:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler
