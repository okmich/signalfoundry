import pytest

from okmich_quant_neural_net.keras.paper2keras.wavenet_causal_cnn import (
    build_wavenet_cnn,
    build_wavenet_cnn_tunable,
)


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


def test_build_wavenet_cnn_raises_clear_error_when_filters_and_dilations_mismatch():
    with pytest.raises(ValueError, match=r"len\(filters_list\)=4 and len\(dilations\)=5"):
        build_wavenet_cnn(
            sequence_length=32,
            num_features=8,
            num_classes=3,
            filters_list=(32, 32, 64, 64),
            dilations=(1, 2, 4, 8, 16),
        )


def test_build_wavenet_cnn_raises_clear_error_for_skip_with_non_uniform_filters():
    with pytest.raises(ValueError, match="use_skip_connections=True"):
        build_wavenet_cnn(
            sequence_length=32,
            num_features=8,
            num_classes=3,
            filters_list=(32, 32, 64, 64),
            dilations=(1, 2, 4, 8),
            use_skip_connections=True,
        )


@pytest.mark.parametrize("dilation_strategy,expected_length", [("medium", 5), ("deep", 6)])
def test_build_wavenet_cnn_tunable_aligns_filters_to_dilation_length(dilation_strategy, expected_length):
    hp = DummyHyperParameters(
        choices={
            "kernel_size": 2,
            "filter_strategy": "medium",
            "dilation_strategy": dilation_strategy,
            "dense1_units": 64,
            "dense2_units": 32,
            "optimizer": "adam",
        },
        floats={
            "dropout_rate": 0.2,
            "dense1_dropout": 0.3,
            "dense2_dropout": 0.2,
            "learning_rate": 1e-3,
        },
        booleans={"use_skip_connections": False, "use_batch_norm": False},
    )

    model = build_wavenet_cnn_tunable(
        hp=hp,
        num_features=8,
        num_classes=3,
        sequence_length=48,
    )

    tcn_layer = model.get_layer("wavenet_tcn")
    assert isinstance(tcn_layer.nb_filters, list)
    assert len(tcn_layer.nb_filters) == expected_length
    assert len(tcn_layer.dilations) == expected_length


def test_build_wavenet_cnn_tunable_disables_skip_connections_for_non_uniform_filters():
    hp = DummyHyperParameters(
        choices={
            "kernel_size": 2,
            "filter_strategy": "large",
            "dilation_strategy": "deep",
            "dense1_units": 64,
            "dense2_units": 32,
            "optimizer": "adam",
        },
        floats={
            "dropout_rate": 0.2,
            "dense1_dropout": 0.3,
            "dense2_dropout": 0.2,
            "learning_rate": 1e-3,
        },
        booleans={"use_skip_connections": True, "use_batch_norm": False},
    )

    model = build_wavenet_cnn_tunable(
        hp=hp,
        num_features=8,
        num_classes=3,
        sequence_length=48,
    )

    tcn_layer = model.get_layer("wavenet_tcn")
    assert tcn_layer.use_skip_connections is False
