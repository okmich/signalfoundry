import numpy as np
import pytest
import keras
from keras import layers as keras_layers

from okmich_quant_neural_net.keras.layers import HierarchicalCellAttention


def _np(tensor):
    return keras.ops.convert_to_numpy(tensor)


class TestHierarchicalCellAttentionInitialization:

    def test_defaults(self):
        layer = HierarchicalCellAttention()
        assert layer.num_heads == 4 and layer.temperature == 1.0

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            HierarchicalCellAttention(num_heads=0)
        with pytest.raises(ValueError):
            HierarchicalCellAttention(temperature=0.0)

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError):
            HierarchicalCellAttention()(np.zeros((2, 10), dtype=np.float32))

    def test_rejects_dynamic_rows_or_cols(self):
        with pytest.raises(ValueError, match="static"):
            HierarchicalCellAttention()(keras_layers.Input(shape=(None, 10)))
        with pytest.raises(ValueError, match="static"):
            HierarchicalCellAttention()(keras_layers.Input(shape=(14, None)))

    def test_weight_shapes_and_count(self):
        layer = HierarchicalCellAttention(num_heads=4)
        layer.build((None, 14, 10))
        assert tuple(layer.w_cell.shape) == (14, 10)
        assert tuple(layer.w_row.shape) == (14, 4)
        assert layer.count_params() == 14 * 10 + 14 * 4

    def test_get_config_round_trip(self):
        layer = HierarchicalCellAttention(num_heads=5, temperature=0.7, name="hca")
        clone = HierarchicalCellAttention.from_config(layer.get_config())
        assert clone.num_heads == 5 and clone.temperature == 0.7 and clone.name == "hca"


class TestHierarchicalCellAttentionForward:

    def test_output_shapes(self):
        x = np.random.default_rng(0).random((5, 14, 10), dtype=np.float32)
        out, attn = HierarchicalCellAttention(num_heads=4)(x)
        assert tuple(out.shape) == (5, 4, 10)
        assert tuple(attn.shape) == (5, 14, 4)

    def test_compute_output_shape(self):
        shapes = HierarchicalCellAttention(num_heads=3).compute_output_shape((None, 8, 10))
        assert shapes == ((None, 3, 10), (None, 8, 3))

    def test_attention_sums_to_one_per_head(self):
        _, attn = HierarchicalCellAttention(num_heads=3)(np.random.default_rng(1).random((7, 8, 10), dtype=np.float32))
        np.testing.assert_allclose(_np(attn).sum(axis=1), 1.0, atol=1e-5)

    def test_zero_input_gives_uniform_attention_and_zero_output(self):
        out, attn = HierarchicalCellAttention(num_heads=2)(np.zeros((3, 14, 10), dtype=np.float32))
        np.testing.assert_allclose(_np(attn), 1.0 / 14, atol=1e-6)
        np.testing.assert_allclose(_np(out), 0.0, atol=1e-6)

    def test_matches_numpy_reference_of_paper_equations(self):
        rng = np.random.default_rng(11)
        batch, rows, cols, heads, tau = 5, 14, 10, 4, 0.7
        x = rng.random((batch, rows, cols)).astype(np.float32)
        w_cell = rng.normal(size=(rows, cols)).astype(np.float32)
        w_row = rng.normal(size=(rows, heads)).astype(np.float32)
        layer = HierarchicalCellAttention(num_heads=heads, temperature=tau)
        layer.build((None, rows, cols))
        layer.set_weights([w_cell, w_row])
        out, attn = layer(x)
        row_scores = (x * w_cell).sum(-1)  # eq. (1) aggregated per row
        head_scores = row_scores[:, :, None] * w_row[None]  # score per row and head
        z = head_scores / tau
        z -= z.max(axis=1, keepdims=True)
        ref_attn = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)  # eq. (2)
        ref_out = np.einsum("brh,brf->bhf", ref_attn, x)
        np.testing.assert_allclose(_np(attn), ref_attn, atol=1e-5)
        np.testing.assert_allclose(_np(out), ref_out, atol=1e-5)

    def test_lower_temperature_is_sharper(self):
        x = np.random.default_rng(3).random((16, 14, 10), dtype=np.float32)
        sharp = HierarchicalCellAttention(num_heads=1, temperature=0.1)
        soft = HierarchicalCellAttention(num_heads=1, temperature=10.0)
        sharp.build(x.shape)
        soft.build(x.shape)
        soft.set_weights(sharp.get_weights())
        _, a_sharp = sharp(x)
        _, a_soft = soft(x)
        assert _np(a_sharp).max(axis=1).mean() > _np(a_soft).max(axis=1).mean()

    def test_trains_inside_a_functional_model(self):
        inputs = keras_layers.Input(shape=(14, 10))
        out, _ = HierarchicalCellAttention(num_heads=2, name="hca")(inputs)
        logits = keras_layers.Dense(3, activation="softmax")(keras_layers.Flatten()(out))
        model = keras.Model(inputs, logits)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        rng = np.random.default_rng(4)
        before = [w.copy() for w in model.get_layer("hca").get_weights()]
        model.fit(rng.random((64, 14, 10), dtype=np.float32), rng.integers(0, 3, size=64), epochs=2, batch_size=16,
                  verbose=0)
        after = model.get_layer("hca").get_weights()
        assert all(not np.allclose(b, a) for b, a in zip(before, after))
