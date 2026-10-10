import keras_tuner as kt
import numpy as np
import pytest
import keras

from okmich_quant_neural_net.keras.layers import HierarchicalCellAttention
from okmich_quant_neural_net.keras.paper2keras.common import TaskType
from okmich_quant_neural_net.keras.paper2keras.taan import (
    AttentionOutput, TaanLayer, build_attention_model, create_taan, create_tunable_taan, default_tuner_objective,
)

PAPER_COLS = 10
PAPER_OSCILLATORS = ("rsi", "er", "ift_rsi", "adx", "stoch", "stoch_d", "di_plus", "di_minus")
PAPER_SHAPES = {"zigzag": (8, PAPER_COLS), **{name: (14, PAPER_COLS) for name in PAPER_OSCILLATORS}}
PAPER_PARAMS = 26_006
SMALL_SHAPES = {"rsi": (14, 10), "zigzag": (8, 10)}


def _inputs(shapes, n, seed=0):
    rng = np.random.default_rng(seed)
    return {name: rng.random((n, rows, cols), dtype=np.float32) for name, (rows, cols) in shapes.items()}


class TestCreateTaanValidation:

    def test_rejects_mismatched_columns(self):
        with pytest.raises(ValueError, match="same number of columns"):
            create_taan({"rsi": (14, 10), "zigzag": (8, 9)}, num_classes=3)

    def test_rejects_empty_mapping(self):
        with pytest.raises(ValueError, match="at least one"):
            create_taan({}, num_classes=3)

    @pytest.mark.parametrize("bad", [str(TaanLayer.DENSE_1), str(TaanLayer.LEVEL2), f"{TaanLayer.LEVEL1_PREFIX}rsi",
                                     f"{TaanLayer.LEVEL1_FLAT_PREFIX}rsi", str(TaanLayer.DROPOUT_1)])
    def test_rejects_reserved_indicator_names(self, bad):
        with pytest.raises(ValueError, match="collides"):
            create_taan({bad: (14, 10), "zigzag": (8, 10)}, num_classes=3)

    @pytest.mark.parametrize("bad", ["", "di+", "stoch %k", "(zz)", "a/b", 7])
    def test_rejects_names_that_are_not_valid_layer_names(self, bad):
        with pytest.raises(ValueError, match="indicator name"):
            create_taan({bad: (14, 10), "zigzag": (8, 10)}, num_classes=3)

    def test_accepts_names_with_dot_dash_underscore(self):
        shapes = {"stoch.k": (14, 10), "di-minus": (14, 10), "ift_rsi": (14, 10)}
        model = create_taan(shapes, num_classes=3)
        assert model.predict(_inputs(shapes, 2), verbose=0).shape == (2, 3)

    @pytest.mark.parametrize("bad_shape", [(14, 10, 1), (14,), 14, (0, 10), (14, -1), (14.0, 10), [14, 10]])
    def test_rejects_malformed_shapes(self, bad_shape):
        with pytest.raises(ValueError, match="matrix_shapes"):
            create_taan({"rsi": bad_shape, "zigzag": (8, 10)}, num_classes=3)

    @pytest.mark.parametrize("bad", [None, 1, 0, -3, 2.0, "3"])
    def test_rejects_invalid_num_classes_for_classification(self, bad):
        with pytest.raises(ValueError, match="num_classes"):
            create_taan(SMALL_SHAPES, num_classes=bad)
        with pytest.raises(ValueError, match="num_classes"):
            default_tuner_objective(bad)

    def test_rejects_unsupported_task_type(self):
        with pytest.raises(ValueError, match="supports"):
            create_taan(SMALL_SHAPES, num_classes=3, task_type=TaskType.POSTERIOR_DISTILLATION)
        with pytest.raises(ValueError, match="supports"):
            default_tuner_objective(3, TaskType.POSTERIOR_DISTILLATION)

    def test_inputs_are_keyed_by_indicator_name(self):
        shapes = {"zigzag": (8, 10), "rsi": (14, 10)}  # insertion order != Keras's sorted flat order
        model = create_taan(shapes, num_classes=3)
        assert {tensor.name for tensor in model.inputs} == set(shapes)
        x = _inputs(shapes, 4)
        # Dict feeding binds by name regardless of key order.
        np.testing.assert_allclose(model.predict({"rsi": x["rsi"], "zigzag": x["zigzag"]}, verbose=0),
                                   model.predict({"zigzag": x["zigzag"], "rsi": x["rsi"]}, verbose=0), atol=1e-6)


class TestCreateTaanArchitecture:

    def test_paper_configuration_parameter_count(self):
        model = create_taan(PAPER_SHAPES, num_classes=3)
        assert model.count_params() == PAPER_PARAMS
        by_layer = {layer.name: layer.count_params() for layer in model.layers if layer.count_params()}
        assert by_layer[f"{TaanLayer.LEVEL1_PREFIX}zigzag"] == 8 * 10 + 8 * 4
        assert by_layer[f"{TaanLayer.LEVEL1_PREFIX}rsi"] == 14 * 10 + 14 * 4
        assert by_layer[str(TaanLayer.LEVEL2)] == 9 * 40 + 9 * 3
        assert by_layer[str(TaanLayer.DENSE_1)] == 120 * 128 + 128

    def test_forward_shapes_mixed_rows(self):
        model = create_taan(PAPER_SHAPES, num_classes=3)
        proba = model.predict(_inputs(PAPER_SHAPES, 6), verbose=0)
        assert proba.shape == (6, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_single_matrix_model_keeps_uniform_layer_graph(self):
        shapes = {"rsi": (14, 10)}
        model = create_taan(shapes, num_classes=3)
        assert model.predict(_inputs(shapes, 4), verbose=0).shape == (4, 3)
        assert model.get_layer(str(TaanLayer.LEVEL2_STACK)).output.shape == (None, 1, 40)
        attn = build_attention_model(model).predict(_inputs(shapes, 4), verbose=0)
        np.testing.assert_allclose(attn[str(AttentionOutput.LEVEL2)], 1.0)  # one indicator: trivially selected

    def test_level2_rows_follow_insertion_order(self):
        shapes = {"zigzag": (8, 10), "rsi": (14, 10), "stoch": (14, 10)}  # insertion order != alphabetical
        model = create_taan(shapes, num_classes=3)
        x = _inputs(shapes, 5)
        stack = keras.models.Model(inputs={t.name: t for t in model.inputs},
                                   outputs=model.get_layer(str(TaanLayer.LEVEL2_STACK)).output)
        base = stack.predict(x, verbose=0)
        assert base.shape == (5, 3, 40)
        for position, name in enumerate(shapes):
            perturbed = stack.predict({**x, name: np.zeros_like(x[name])}, verbose=0)
            changed_rows = np.where(np.abs(base - perturbed).max(axis=(0, 2)) > 0)[0]
            assert changed_rows.tolist() == [position]

    def test_dropout_and_l2_layers_present_when_requested(self):
        model = create_taan(SMALL_SHAPES, num_classes=3, dense_dropout=0.2, l2_reg=1e-4)
        names = {layer.name for layer in model.layers}
        assert {str(TaanLayer.DROPOUT_1), str(TaanLayer.DROPOUT_2)} <= names
        assert model.get_layer(str(TaanLayer.DENSE_1)).kernel_regularizer is not None


class TestCreateTaanHeads:

    def test_multiclass_metrics_and_class_weight(self):
        shapes = {"rsi": (14, 10)}
        model = create_taan(shapes, num_classes=3)
        y = np.random.default_rng(5).integers(0, 3, size=32)
        history = model.fit(_inputs(shapes, 32), y, epochs=1, batch_size=16, verbose=0,
                            class_weight={0: 1.0, 1: 3.0, 2: 3.0})
        assert {"loss", "accuracy", "macro_f1", "balanced_accuracy"} <= set(history.history)
        assert default_tuner_objective(3) == ("val_macro_f1", "max")

    def test_binary_head_and_metrics(self):
        model = create_taan(SMALL_SHAPES, num_classes=2)
        x = _inputs(SMALL_SHAPES, 16)
        assert model.predict(x, verbose=0).shape == (16, 1)
        history = model.fit(x, np.random.default_rng(6).integers(0, 2, size=16), epochs=1, batch_size=8, verbose=0)
        assert {"loss", "accuracy", "precision", "recall", "auc"} <= set(history.history)
        assert default_tuner_objective(2) == ("val_auc", "max")

    def test_regression_head(self):
        model = create_taan({"rsi": (14, 10)}, num_classes=None, task_type=TaskType.REGRESSION)
        assert model.predict(_inputs({"rsi": (14, 10)}, 4), verbose=0).shape == (4, 1)
        assert default_tuner_objective(None, TaskType.REGRESSION) == ("val_loss", "min")

    def test_training_updates_every_attention_weight(self):
        model = create_taan(SMALL_SHAPES, num_classes=3)
        x = _inputs(SMALL_SHAPES, 64)
        y = np.random.default_rng(4).integers(0, 3, size=64)
        attention_layers = [layer for layer in model.layers if isinstance(layer, HierarchicalCellAttention)]
        assert len(attention_layers) == len(SMALL_SHAPES) + 1
        before = {layer.name: [w.copy() for w in layer.get_weights()] for layer in attention_layers}
        model.fit(x, y, epochs=2, batch_size=16, verbose=0)
        for layer in attention_layers:
            for b, a in zip(before[layer.name], layer.get_weights()):
                assert not np.allclose(b, a), layer.name


class TestAttentionModel:

    def test_outputs_shapes_and_normalisation(self):
        model = create_taan(PAPER_SHAPES, num_classes=3, level1_heads=4, level2_heads=3)
        attention = build_attention_model(model)
        assert set(tensor.name for tensor in attention.inputs) == set(PAPER_SHAPES)
        attn = attention.predict(_inputs(PAPER_SHAPES, 5), verbose=0)
        expected_keys = {f"{AttentionOutput.LEVEL1_PREFIX}{name}" for name in PAPER_SHAPES}
        expected_keys.add(str(AttentionOutput.LEVEL2))
        assert set(attn) == expected_keys
        assert attn[f"{AttentionOutput.LEVEL1_PREFIX}zigzag"].shape == (5, 8, 4)
        assert attn[f"{AttentionOutput.LEVEL1_PREFIX}rsi"].shape == (5, 14, 4)
        assert attn[str(AttentionOutput.LEVEL2)].shape == (5, len(PAPER_SHAPES), 3)
        for value in attn.values():
            np.testing.assert_allclose(value.sum(axis=1), 1.0, atol=1e-5)

    def test_level1_attention_is_per_indicator(self):
        shapes = {"zigzag": (8, 10), "rsi": (14, 10), "stoch": (14, 10)}
        attention = build_attention_model(create_taan(shapes, num_classes=3))
        x = _inputs(shapes, 5)
        base = attention.predict(x, verbose=0)
        perturbed = attention.predict({**x, "stoch": np.zeros_like(x["stoch"])}, verbose=0)
        np.testing.assert_allclose(perturbed[f"{AttentionOutput.LEVEL1_PREFIX}zigzag"],
                                   base[f"{AttentionOutput.LEVEL1_PREFIX}zigzag"])
        np.testing.assert_allclose(perturbed[f"{AttentionOutput.LEVEL1_PREFIX}stoch"], 1.0 / 14, atol=1e-6)

    def test_rejects_non_taan_model_and_names_found_layers(self):
        inputs = keras.layers.Input(shape=(4,))
        with pytest.raises(ValueError, match="missing level-2 layer"):
            build_attention_model(keras.Model(inputs, keras.layers.Dense(1)(inputs)))
        matrix = keras.layers.Input(shape=(14, 10))
        out, _ = HierarchicalCellAttention(name="hand_built")(matrix)
        with pytest.raises(ValueError, match="hand_built"):
            build_attention_model(keras.Model(matrix, keras.layers.Flatten()(out)))


class TestSerialisation:

    def test_model_config_round_trip(self):
        model = create_taan(SMALL_SHAPES, num_classes=3)
        x = _inputs(SMALL_SHAPES, 8)
        clone = keras.models.Model.from_config(model.get_config())
        clone.set_weights(model.get_weights())
        np.testing.assert_allclose(clone.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6)
        clone_attn = build_attention_model(clone).predict(x, verbose=0)
        model_attn = build_attention_model(model).predict(x, verbose=0)
        assert set(clone_attn) == set(model_attn)
        for key in model_attn:
            np.testing.assert_allclose(clone_attn[key], model_attn[key], atol=1e-6)

    def test_compile_config_round_trip_and_clone_model(self):
        # clone_model re-compiles from the serialized compile config, which includes the custom metrics.
        model = create_taan(SMALL_SHAPES, num_classes=3)
        x = _inputs(SMALL_SHAPES, 8)
        clone = keras.models.clone_model(model)
        clone.set_weights(model.get_weights())
        np.testing.assert_allclose(clone.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6)
        assert clone.compiled
        rebuilt = keras.models.Model.from_config(model.get_config())
        rebuilt.compile_from_config(model.get_compile_config())
        y = np.random.default_rng(7).integers(0, 3, size=8)
        names = set(rebuilt.evaluate(x, y, verbose=0, return_dict=True))
        assert {"loss", "accuracy", "macro_f1", "balanced_accuracy"} <= names

    def test_save_and_load_round_trip(self, tmp_path):
        import h5py
        if not hasattr(h5py, "File"):
            pytest.skip("h5py is not importable as a real package in this environment (broken install)")
        model = create_taan(SMALL_SHAPES, num_classes=3)
        x = _inputs(SMALL_SHAPES, 8)
        expected = model.predict(x, verbose=0)
        path = tmp_path / "taan.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        np.testing.assert_allclose(loaded.predict(x, verbose=0), expected, atol=1e-6)
        np.testing.assert_allclose(build_attention_model(loaded).predict(x, verbose=0)[str(AttentionOutput.LEVEL2)],
                                   build_attention_model(model).predict(x, verbose=0)[str(AttentionOutput.LEVEL2)],
                                   atol=1e-6)


class TestTunable:

    def test_tunable_builds_with_real_hyperparameters_and_registers_search_space(self):
        hp = kt.HyperParameters()
        model = create_tunable_taan(SMALL_SHAPES, num_classes=3)(hp)
        assert model.predict(_inputs(SMALL_SHAPES, 3), verbose=0).shape == (3, 3)
        expected = {"level1_heads", "level2_heads", "temperature", "dense_units_1", "dense_units_2", "dense_dropout",
                    "learning_rate", "l2_reg"}
        assert expected <= set(hp.values)
        assert model.get_layer(str(TaanLayer.LEVEL2)).num_heads == hp.values["level2_heads"]
        assert model.get_layer(str(TaanLayer.DENSE_1)).units == hp.values["dense_units_1"]

    def test_tunable_binary_logs_its_objective(self):
        model = create_tunable_taan(SMALL_SHAPES, num_classes=2)(kt.HyperParameters())
        x = _inputs(SMALL_SHAPES, 16)
        y = np.random.default_rng(8).integers(0, 2, size=16)
        history = model.fit(x, y, validation_data=(x, y), epochs=1, batch_size=8, verbose=0)
        assert default_tuner_objective(2)[0] in history.history
