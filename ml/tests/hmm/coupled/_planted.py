"""A planted direction x volatility coupled model shared by the tests."""
import numpy as np

from okmich_quant_ml.hmm.coupled import CoupledHMM, GaussianEmission, LocationScaleEmission

DIR, VOL = 0, 1
FULL = [(DIR, VOL), (VOL, DIR)]
NONE = [(DIR,), (VOL,)]


def direction_vol_model(parents, **kwargs) -> CoupledHMM:
    emissions = [LocationScaleEmission(0, location=DIR, scale=VOL), GaussianEmission(1, chain=VOL)]
    return CoupledHMM((3, 2), emissions, parents, **kwargs)


def planted(parents=FULL, burst_switch: float = 0.08, calm_switch: float = 0.01) -> CoupledHMM:
    """Direction switches far more often in bursts; bear raises the calm -> burst hazard."""
    model = direction_vol_model(parents)
    model._init_params(np.zeros((10, 2)) + np.arange(10)[:, None], np.random.default_rng(0), 0)
    a_d = np.empty((3, 2, 3))
    for d in range(3):
        for v, p in enumerate((calm_switch, burst_switch)):
            a_d[d, v] = p / 2
            a_d[d, v, d] = 1 - p
    a_v = np.empty((3, 2, 2))
    for d, enter in enumerate((0.04, 0.02, 0.01)):
        a_v[d, 0] = (1 - enter, enter)
        a_v[d, 1] = (0.08, 0.92)
    model.transitions = [a_d, a_v] if parents == FULL else [a_d.mean(axis=1), a_v.mean(axis=0)]
    model.pi = np.full(6, 1 / 6)
    model.emissions[0].mu = np.array([-0.4, 0.0, 0.4])
    model.emissions[0].scale = np.array([0.7, 1.6])
    model.emissions[1].mean = np.array([-0.6, 0.9])
    model.emissions[1].var = np.array([0.25, 0.3])
    return model
