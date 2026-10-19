import keras
import numpy as np
import pytest

from okmich_quant_neural_net.keras.paper2keras.cnn_rnn_hybrid import build_cnn_rnn, build_deep_cnn_rnn, \
    build_cnn_rnn_tunable
from okmich_quant_neural_net.keras.paper2keras.common import causal_max_pooling_1d

NUM_FEATURES = 8


class DummyHyperParameters:
    def __init__(self, choices, floats, booleans):
        self._choices = choices
        self._floats = floats
        self._booleans = booleans

    def Choice(self, name, values):
        value = self._choices[name]
        assert value in values
        return value

    def Float(self, name, min_value=None, max_value=None, step=None, sampling=None):
        value = self._floats[name]
        if min_value is not None:
            assert value >= min_value
        if max_value is not None:
            assert value <= max_value
        return value

    def Boolean(self, name, default=False):
        return self._booleans.get(name, default)


def _build_simple(sequence_length, pool_size):
    return build_cnn_rnn(sequence_length=sequence_length, num_features=NUM_FEATURES, num_classes=3, pool_size=pool_size)


def _build_deep(sequence_length, pool_size):
    return build_deep_cnn_rnn(sequence_length=sequence_length, num_features=NUM_FEATURES, num_classes=3,
                              pool_size=pool_size)


def _build_tunable(sequence_length, pool_size):
    choices = {"conv1_filters": 32, "conv2_filters": 32, "conv_kernel_size": 3, "pool_size": pool_size,
               "rnn_units": 32, "dense1_units": 64, "dense2_units": 32, "optimizer": "adam"}
    floats = {"dropout1": 0.2, "dropout2": 0.1, "learning_rate": 1e-3}
    hp = DummyHyperParameters(choices, floats, {"use_batch_norm": True, "use_residual": False})
    return build_cnn_rnn_tunable(hp, num_features=NUM_FEATURES, num_classes=3, sequence_length=sequence_length)


BUILDERS = [_build_simple, _build_deep, _build_tunable]


def _output_moves(model, sequence_length, timesteps):
    """True if perturbing the given timesteps of a random input changes the model output."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, sequence_length, NUM_FEATURES)).astype("float32")
    x_perturbed = x.copy()
    x_perturbed[:, timesteps, :] += rng.normal(scale=5.0, size=(2, len(timesteps), NUM_FEATURES))
    base = keras.ops.convert_to_numpy(model(x, training=False))
    moved = keras.ops.convert_to_numpy(model(x_perturbed, training=False))
    return not np.allclose(base, moved, atol=1e-7)


@pytest.fixture(autouse=True)
def _seed():
    keras.utils.set_random_seed(0)


@pytest.mark.parametrize("build", BUILDERS)
@pytest.mark.parametrize("sequence_length,pool_size", [(7, 2), (75, 2), (50, 3), (100, 3)])
def test_newest_bar_reaches_the_output_when_length_is_not_a_pool_multiple(build, sequence_length, pool_size):
    model = build(sequence_length, pool_size)
    assert _output_moves(model, sequence_length, [sequence_length - 1])


@pytest.mark.parametrize("sequence_length,pool_size,expected", [(7, 2, [2, 4, 6]), (8, 3, [4, 7])])
def test_causal_max_pooling_windows_end_on_the_newest_bar(sequence_length, pool_size, expected):
    # valid pooling from the oldest bar would give [1, 3, 5] and [2, 5]
    inputs = keras.Input(shape=(sequence_length, 1))
    model = keras.Model(inputs, causal_max_pooling_1d(inputs, pool_size, name="pool"))
    x = np.arange(sequence_length, dtype="float32").reshape(1, sequence_length, 1)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(model(x)).ravel(), expected)


@pytest.mark.parametrize("build", BUILDERS)
def test_no_crop_layer_when_length_is_a_pool_multiple(build):
    model = build(100, 2)
    assert not any(layer.name.endswith("_crop_oldest") for layer in model.layers)
