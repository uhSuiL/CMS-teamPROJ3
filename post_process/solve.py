import numpy as np
from numpy import ndarray
from typing import Optional

from post_process.objective import total_cost


def hard_assign(X: ndarray, out_dtype: np.dtype) -> ndarray:
    """
    Enforce one-label-per-voxel constraint.
    argmax over label dim is equivalent to softmax → argmax (softmax is monotone).

    :param X: (N, W, H, D, L) continuous logits
    :return:  (N, W, H, D, L) one-hot, dtype=out_dtype
    """
    labels = np.argmax(X, axis=-1)                          # (N, W, H, D)
    return np.eye(X.shape[-1], dtype=out_dtype)[labels]     # (N, W, H, D, L)


def solve_sa(
    C_unary: ndarray,
    C_adj: ndarray,
    step_init: float,
    X_init: ndarray = None,
    *,
    T_init: float = 1.0,
    T_final: float = 1e-4,
    n_iter: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> ndarray:
    """
    Black-box simulated annealing for MRF labeling.

    Optimization variable: continuous logit tensor X of shape (N, W, H, D, L).
    Constraint: exactly one label per voxel, enforced via argmax at each
    evaluation — equivalent to softmax → argmax.

    The objective is treated as a black box:
        cost = total_cost(C_unary, hard_assign(X_proposed), C_adj)
    No knowledge of the internal structure of total_cost is assumed.

    Perturbation: isotropic Gaussian noise on X, std = current temperature.
    Accept / reject independently per sample via the Metropolis criterion.
    Temperature decays geometrically from T_init to T_final over n_iter steps.

    Tip — T_init should be on the same order of magnitude as a typical cost
    difference between two labelings (e.g. inspect total_cost on a few random
    hard assignments to calibrate).

    :param C_unary: (N, W, H, D, L) per-voxel-per-label unary costs
    :param C_adj:   (L, L) pairwise adjacency cost matrix
    :param X_init:
    :param T_init:  initial temperature (controls initial noise std)
    :param T_final: final temperature
    :param n_iter:  number of SA iterations
    :param rng:     numpy Generator for reproducibility
    :return:        (N, W, H, D, L) one-hot label assignment
    """
    if rng is None:
        rng = np.random.default_rng()

    N, W, H, D, L = C_unary.shape
    orig_dtype = C_unary.dtype

    # ── Initialize logits ─────────────────────────────────────────────────────
    # X = −C_unary: low-cost labels get high logits → argmax recovers the
    # unary-greedy solution as the starting point.
    X = -C_unary.astype(np.float64) if X_init is None else X_init  # float64 for numerical stability

    cost = total_cost(C_unary, hard_assign(X, orig_dtype), C_adj)  # (N, 1)

    T = float(T_init)
    decay = (T_final / T_init) ** (1.0 / max(n_iter - 1, 1))

    for i in range(n_iter):
        # ── Propose ───────────────────────────────────────────────────────────
        # X_prop = X + rng.standard_normal(X.shape) * T
        step = step_init * (T / T_init)
        X_prop = X + rng.uniform(-step, step, size=X.shape)

        cost_prop = total_cost(
            C_unary, hard_assign(X_prop, orig_dtype), C_adj
        )  # (N, 1)

        # ── Metropolis accept / reject (per sample) ───────────────────────────
        delta = cost_prop - cost                              # (N, 1)
        log_u = np.log(rng.uniform(size=(N, 1)) + 1e-20)
        accept = (delta <= 0) | (log_u < -delta / T)          # (N, 1) bool

        X    = np.where(accept.reshape(N, 1, 1, 1, 1), X_prop, X)
        cost = np.where(accept, cost_prop, cost)

        T *= decay

        print(f"Iter {i} | Cost: {cost_prop.mean():.4f}")

    return hard_assign(X, orig_dtype)
