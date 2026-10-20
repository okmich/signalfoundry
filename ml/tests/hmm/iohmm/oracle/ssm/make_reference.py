"""
Reference fixture for ``../../test_ssm_oracle.py``, computed by ssm (lindermanlab/ssm) inside the container built
from ``Dockerfile``.

ssm's input-driven HMM gives each destination state one weight vector, shared by every origin::

    P(z_t = j | z_{t-1} = i, u_t)  proportional to  exp(log_Ps[i, j] + Ws[j] . u_t)

That is the special case ``W[i, j] = Ws[j] - Ws[i]`` of our per-(origin, destination) model, so every ssm model has an
exact counterpart in ours. Sequences are sampled from a known ssm model (3 states, 3 diagonal-Gaussian features, 2
persistent inputs); a fresh ssm model is fitted to them by EM, and its parameters and inference outputs are saved.
ssm uses the input on row t for the transition from t - 1 into t (our KNOWN_AHEAD).
"""
import sys

import numpy as np
import numpy.random as npr
import ssm
from ssm.messages import forward_pass

K, D, M = 3, 3, 2
N_SEQ, SEQ_LEN = 4, 3000
INPUT_PERSISTENCE = 0.95
SEED = 11

TRUE_P = np.array([[0.95, 0.03, 0.02], [0.03, 0.94, 0.03], [0.02, 0.03, 0.95]])
TRUE_WS = np.array([[1.0, 0.0], [-0.5, 0.8], [0.0, -1.0]])  # (K, M): input m adds Ws[j, m] to destination j's logit
TRUE_MUS = np.array([[-1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [1.0, -1.0, 0.0]])
TRUE_SIGMASQ = np.array([[0.6, 1.0, 0.8], [1.0, 0.5, 1.2], [0.8, 1.2, 0.6]])


def persistent_inputs(n_bars: int) -> np.ndarray:
    """Unit-variance AR(1) inputs, one column per input."""
    u = np.zeros((n_bars, M))
    u[0] = npr.randn(M)
    scale = np.sqrt(1 - INPUT_PERSISTENCE ** 2)
    for t in range(1, n_bars):
        u[t] = INPUT_PERSISTENCE * u[t - 1] + scale * npr.randn(M)
    return u


def main(out_path: str) -> None:
    npr.seed(SEED)  # ssm samples and initialises with numpy's global generator
    true = ssm.HMM(K, D, M=M, observations="diagonal_gaussian", transitions="inputdriven")
    true.init_state_distn.log_pi0 = np.log(np.full(K, 1.0 / K))
    true.transitions.log_Ps = np.log(TRUE_P)
    true.transitions.Ws = TRUE_WS.copy()
    true.observations.mus = TRUE_MUS.copy()
    true.observations.sigmasq = TRUE_SIGMASQ.copy()

    us, zs, xs = [], [], []
    for _ in range(N_SEQ):
        u = persistent_inputs(SEQ_LEN)
        z, x = true.sample(SEQ_LEN, input=u)
        us.append(u)
        zs.append(z)
        xs.append(x)

    model = ssm.HMM(K, D, M=M, observations="diagonal_gaussian", transitions="inputdriven")
    lls = model.fit(xs, inputs=us, method="em", num_iters=1000, tolerance=1e-6, verbose=0)

    pi0 = model.init_state_distn.initial_state_distn
    out = {name: [] for name in ("Ps", "log_likes", "filtered", "predicted", "smoothed", "viterbi", "loglik")}
    for x, u in zip(xs, us):
        mask = np.ones_like(x, dtype=bool)
        Ps = model.transitions.transition_matrices(x, u, mask, None)
        log_likes = model.observations.log_likelihoods(x, u, mask, None)
        alphas = np.zeros((len(x), K))
        forward_pass(pi0, Ps, log_likes, alphas)
        filtered = np.exp(alphas - alphas.max(axis=1, keepdims=True))
        smoothed, _, normalizer = model.expected_states(x, input=u)
        out["Ps"].append(Ps)
        out["log_likes"].append(log_likes)
        out["filtered"].append(filtered / filtered.sum(axis=1, keepdims=True))
        out["predicted"].append(model.filter(x, input=u))
        out["smoothed"].append(smoothed)
        out["viterbi"].append(model.most_likely_states(x, input=u))
        out["loglik"].append(normalizer)

    np.savez_compressed(
        out_path, x=np.stack(xs), u=np.stack(us), z=np.stack(zs), true_P=TRUE_P, true_Ws=TRUE_WS, true_mus=TRUE_MUS,
        true_sigmasq=TRUE_SIGMASQ, pi0=pi0, log_Ps=model.transitions.log_Ps, Ws=model.transitions.Ws,
        mus=model.observations.mus, sigmasq=model.observations.sigmasq, em_lls=np.asarray(lls),
        total_loglik=model.log_likelihood(xs, inputs=us), **{k: np.stack(v) for k, v in out.items()})
    print(f"saved {out_path}: EM iterations {len(lls)}, log-likelihood {model.log_likelihood(xs, inputs=us):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
