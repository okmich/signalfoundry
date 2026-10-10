from okmich_quant_neural_net.keras.paper2keras.hierarchical_multiscale_cnn import (
    create_hierarchical_multiscale_cnn,
    create_tunable_hierarchical_multiscale_cnn,
)


class DummyHyperParameters:
    def __init__(self, choices, floats, booleans, ints):
        self._choices = choices
        self._floats = floats
        self._booleans = booleans
        self._ints = ints

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

    def Boolean(self, name):
        return self._booleans[name]

    def Int(self, name, min_value=None, max_value=None, step=None):
        value = self._ints[name]
        if min_value is not None:
            assert value >= min_value
        if max_value is not None:
            assert value <= max_value
        return value


def test_create_hierarchical_multiscale_cnn_default_jit_compile_disabled():
    model = create_hierarchical_multiscale_cnn(
        input_shape=(32, 8),
        num_classes=3,
    )

    assert model.jit_compile is False


def test_create_hierarchical_multiscale_cnn_allows_jit_compile_enable():
    model = create_hierarchical_multiscale_cnn(
        input_shape=(32, 8),
        num_classes=3,
        jit_compile=True,
    )

    assert model.jit_compile is True


def test_tunable_builder_propagates_jit_compile_flag():
    builder = create_tunable_hierarchical_multiscale_cnn(
        input_shape=(32, 8),
        num_classes=3,
        jit_compile=False,
    )

    hp = DummyHyperParameters(
        choices={
            "branch_filters": 32,
            "pool_size": 2,
            "conv_filters_1": 128,
            "conv_filters_2": 64,
            "conv_kernel_size": 3,
            "learning_rate": 1e-3,
            "l2_reg": 1e-4,
        },
        floats={
            "conv_dropout": 0.3,
            "dense_dropout_1": 0.3,
            "dense_dropout_2": 0.2,
        },
        booleans={"use_small_kernels": False},
        ints={"dense_units_1": 128, "dense_units_2": 64},
    )

    model = builder(hp)
    assert model.jit_compile is False
