import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from post_process.objective import total_cost


def hard_assign(X: Tensor, out_dtype: torch.dtype) -> Tensor:
    """
    Enforce one-label-per-voxel constraint.
    argmax over label dim is equivalent to softmax → argmax (softmax is monotone).

    :param X:        (N, W, H, D, L) continuous logits
    :param out_dtype: torch dtype for the output one-hot tensor
    :return:         (N, W, H, D, L) one-hot, dtype=out_dtype
    """
    labels = X.argmax(dim=-1)                                       # (N, W, H, D)
    return F.one_hot(labels, num_classes=X.shape[-1]).to(out_dtype) # (N, W, H, D, L)


def solve_sa(
    C_unary: Tensor,
    C_adj: Tensor,
    step_init: float,
    X_init: Optional[Tensor] = None,
    *,
    T_init: float = 1.0,
    T_final: float = 1e-4,
    n_iter: int = 200,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Black-box simulated annealing for MRF labeling.

    Optimization variable: continuous logit tensor X of shape (N, W, H, D, L).
    Constraint: exactly one label per voxel, enforced via argmax at each
    evaluation — equivalent to softmax → argmax.

    The objective is treated as a black box:
        cost = total_cost(C_unary, hard_assign(X_proposed), C_adj)
    No knowledge of the internal structure of total_cost is assumed.

    Perturbation: uniform noise on X, magnitude = current step size.
    Accept / reject independently per sample via the Metropolis criterion.
    Temperature decays geometrically from T_init to T_final over n_iter steps.

    :param C_unary:   (N, W, H, D, L) per-voxel-per-label unary costs
    :param C_adj:     (L, L) pairwise adjacency cost matrix
    :param step_init: initial perturbation magnitude
    :param X_init:    optional warm-start logits; defaults to −C_unary
    :param T_init:    initial temperature
    :param T_final:   final temperature
    :param n_iter:    number of SA iterations
    :param generator: torch Generator for reproducibility
    :return:          (N, W, H, D, L) one-hot label assignment
    """
    device = C_unary.device
    orig_dtype = C_unary.dtype

    N, W, H, D, L = C_unary.shape

    # X = −C_unary: low-cost labels get high logits → argmax recovers the
    # unary-greedy solution as the starting point.
    X = -C_unary.to(torch.float64) if X_init is None else X_init.to(torch.float64)

    cost = total_cost(C_unary, hard_assign(X, orig_dtype), C_adj)  # (N, 1)

    T = float(T_init)
    decay = (T_final / T_init) ** (1.0 / max(n_iter - 1, 1))

    for i in range(n_iter):
        step = step_init * (T / T_init)
        noise = torch.empty_like(X).uniform_(-step, step, generator=generator)
        X_prop = X + noise

        cost_prop = total_cost(
            C_unary, hard_assign(X_prop, orig_dtype), C_adj
        )  # (N, 1)

        # Metropolis accept / reject (per sample)
        delta = cost_prop - cost                                        # (N, 1)
        log_u = torch.log(
            torch.rand((N, 1), dtype=torch.float64, device=device, generator=generator) + 1e-20
        )
        accept = (delta <= 0) | (log_u < -delta / T)                   # (N, 1) bool

        X    = torch.where(accept.reshape(N, 1, 1, 1, 1), X_prop, X)
        cost = torch.where(accept, cost_prop, cost)

        T *= decay

        print(f"Iter {i} | Cost: {cost_prop.mean():.4f}")

    return hard_assign(X, orig_dtype)