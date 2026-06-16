import torch
import torch.nn.functional as F
from torch import Tensor


def labeling_cost(C: Tensor, X: Tensor) -> Tensor:
    """Cost for labeling voxel v as label l, for every v and every l.

    :param C: (N, W, H, D, L)
    :param X: (N, W, H, D, L)
    :return:  (N, 1)
    """
    assert C.shape == X.shape, f'Shape mismatch: {C.shape}, {X.shape}'
    assert C.dim() == X.dim() == 5, f'Dimension mismatch: {C.dim()}, {X.dim()}'
    return (C * X).sum(dim=(1, 2, 3, 4), keepdim=True)


# ─────────────────────────────────────────────
# Adjacency Cost Matrix Builders
# ─────────────────────────────────────────────

def smoothness_penalty(
    num_label: int,
    weight: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: str = None,
) -> Tensor:
    """
    Penalize any adjacent pair of DIFFERENT labels.
    C[l, l'] = weight if l != l' else 0

    :return: (num_label, num_label)
    """
    C = torch.ones((num_label, num_label), dtype=dtype, device=device) * weight
    C.fill_diagonal_(0.0)
    return C


def forbidden_adjacency_penalty(
    num_label: int,
    forbidden_pairs: list[tuple[int, int]],
    weight: float = 1e6,
    dtype: torch.dtype = torch.float32,
    device: str = None,
) -> Tensor:
    """
    Penalize biologically forbidden label adjacencies.

    :return: (num_label, num_label)  (symmetric)
    """
    C = torch.zeros((num_label, num_label), dtype=dtype, device=device)
    for l, l_prime in forbidden_pairs:
        C[l, l_prime] = weight
        C[l_prime, l] = weight
    return C


def combine_adj_costs(*cost_matrices: Tensor) -> Tensor:
    """Additively combine multiple (num_label, num_label) cost matrices."""
    result = torch.zeros_like(cost_matrices[0])
    for C in cost_matrices:
        result = result + C
    return result


# ─────────────────────────────────────────────
# Core Computation
# ─────────────────────────────────────────────

# All 6 axis-aligned neighbor directions in 3D
# Each entry: (axis, shift_direction)
_DIRECTIONS: list[tuple[int, int]] = [
    (0, +1), (0, -1),  # x-axis
    (1, +1), (1, -1),  # y-axis
    (2, +1), (2, -1),  # z-axis
]


def _shift_X(X: Tensor, axis: int, shift: int) -> Tensor:
    """
    Shift X along a spatial axis, padding boundary with zeros.
    Axis 0,1,2 correspond to width, height, depth (spatial dims 1,2,3 in X).

    :param X: (N, W, H, D, L)
    :return:  (N, W, H, D, L)  — neighbor values
    """
    spatial_axis = axis + 1  # offset by 1 due to batch dim
    result = torch.roll(X, shifts=-shift, dims=spatial_axis)

    # zero-out the wrapped boundary (treat boundary as "no neighbor")
    slices = [slice(None)] * X.dim()
    if shift > 0:
        slices[spatial_axis] = slice(-shift, None)   # last `shift` slices
    else:
        slices[spatial_axis] = slice(None, -shift)   # first `|shift|` slices
    result[tuple(slices)] = 0.0
    return result


def adjacency_cost(
    X: Tensor,
    C_adj: Tensor,
    directions: list[tuple[int, int]] = _DIRECTIONS,
) -> Tensor:
    """
    Compute total adjacency cost: Σ_{uv∈E} Σ_{l,l'} C[l,l'] * X[u,l] * X[v,l']

    :param X:     (N, W, H, D, L)
    :param C_adj: (L, L) — shared across all edges
    :param directions: which neighbor directions to include
    :return:      (N, 1)
    """
    assert X.dim() == 5, f'X must be 5D, got {X.dim()}'
    assert C_adj.dim() == 2 and C_adj.shape[0] == C_adj.shape[1] == X.shape[-1]

    C_adj = C_adj.to(dtype=X.dtype, device=X.device)
    total = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

    for axis, shift in directions:
        X_neighbor = _shift_X(X, axis, shift)  # (N, W, H, D, L)
        cost = torch.einsum('nwhdi,ij,nwhdj->n', X, C_adj, X_neighbor)
        total = total + cost

    return total.reshape(-1, 1)


# ─────────────────────────────────────────────
# Total Cost Builder
# ─────────────────────────────────────────────

def total_cost(
    C_unary: Tensor,
    X: Tensor,
    C_adj: Tensor,
    undirected: bool = True,
) -> Tensor:
    """
    Full objective: unary + adjacency terms.

    :param C_unary:    (N, W, H, D, L)
    :param X:          (N, W, H, D, L)
    :param C_adj:      (L, L)
    :param undirected: if True, use only 3 directions to avoid double-counting
    :return:           (N, 1)
    """
    directions = [(0, 1), (1, 1), (2, 1)] if undirected else _DIRECTIONS
    return (
        labeling_cost(C_unary, X)
        + adjacency_cost(X, C_adj, directions=directions)
    )