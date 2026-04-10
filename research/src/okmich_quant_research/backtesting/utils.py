import numpy as np

from okmich_quant_neural_net.tab_2_seq import create_sequences, transform_to_sequence


def generate_seq_feature_func_for_training(
    selected_feature_names,
    feature_engineering_func,
    feature_config,
    sequence_length,
    scaler=None,
):
    def feature_engineering_fn(train_raw, test_raw, train_labels, test_labels):
        # Step 1: Generate 2D features using reusable pipeline
        train_features_2d = feature_engineering_func(train_raw, feature_config)
        test_features_2d = feature_engineering_func(test_raw, feature_config)

        # Step 2: Align labels with features (handle NaN from feature generation)
        train_labels_aligned = train_labels.loc[train_features_2d.index]
        test_labels_aligned = test_labels.loc[test_features_2d.index]

        # Step 3: Select features (same as in main pipeline)
        train_features_2d = train_features_2d[selected_feature_names]
        test_features_2d = test_features_2d[selected_feature_names]

        # Step 4: Convert to numpy arrays
        x_train_2d = train_features_2d.values
        y_train = train_labels_aligned.values
        x_test_2d = test_features_2d.values
        y_test = test_labels_aligned.values

        y_train = np.where(y_train == -1, 0, 1)
        y_test = np.where(y_test == -1, 0, 1)

        # Step 6: Create 3D sequences using transform_to_sequence
        x_train_3d_scaled, y_train_seq = create_sequences(
            features=x_train_2d,
            target=y_train,
            sequence_length=sequence_length,
            scaler=scaler,
            fit_scaler=True,
        )

        x_test_3d_scaled, y_test_seq = create_sequences(
            features=x_test_2d,
            target=y_test,
            sequence_length=sequence_length,
            scaler=scaler,
            fit_scaler=False,
        )
        return x_train_3d_scaled, x_test_3d_scaled, y_train_seq, y_test_seq

    return feature_engineering_fn


def generate_seq_features_for_inference(
    raw_data,
    selected_feature_names,
    feature_engineering_func,
    feature_config,
    sequence_length,
    scaler,
):
    train_features_2d = feature_engineering_func(raw_data, feature_config)
    train_features_2d = train_features_2d[selected_feature_names]
    x_train_2d = train_features_2d.values

    return transform_to_sequence(
        features=x_train_2d, sequence_length=sequence_length, scaler=scaler
    )
