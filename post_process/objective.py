import numpy as np
from numpy import ndarray


def labeling_cost(C: ndarray, X: ndarray) -> ndarray:
    """Cost for Labeling voxel v as label l, for every v and every l

    :param C: (num_sample, width, height, depth, num_label)
    :param X: (num_sample, width, height, depth, num_label)
    :return: (num_sample, 1)
    """
    # sanity check
    assert C.shape == X.shape, f'Shape Mismatch: {C.shape}, {X.shape}'
    assert len(C.shape) == len(X.shape) == 5, f'Dimension Mismatch: {C.shape}, {X.shape}'

    return (C * X).sum(axis=(1, 2, 3, 4), keepdims=True)

# ─────────────────────────────────────────────
# Adjacency Cost Matrix Builders
# ─────────────────────────────────────────────

def smoothness_penalty(num_label: int, weight: float = 1.0) -> ndarray:
    """
    Penalize any adjacent pair of DIFFERENT labels.
    C[l, l'] = weight if l != l' else 0

    :return: (num_label, num_label)
    """
    C = np.ones((num_label, num_label), dtype=np.float32) * weight
    np.fill_diagonal(C, 0.0)
    return C


def forbidden_adjacency_penalty(
    num_label: int,
    forbidden_pairs: list[tuple[int, int]],
    weight: float = 1e6
) -> ndarray:
    """
    Penalize biologically forbidden label adjacencies.
    E.g. forbidden_pairs=[(0, 2)] means label 0 (inner) and label 2 (outer)
    must never be adjacent → assign very high cost.

    :return: (num_label, num_label)  (symmetric)
    """
    C = np.zeros((num_label, num_label), dtype=np.float32)
    for l, l_prime in forbidden_pairs:
        C[l, l_prime] = weight
        C[l_prime, l] = weight
    return C


def combine_adj_costs(*cost_matrices: ndarray) -> ndarray:
    """
    Additively combine multiple (num_label, num_label) cost matrices.
    Since costs are additive, different penalty types compose naturally.
    """
    result = np.zeros_like(cost_matrices[0])
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


def _shift_X(X: ndarray, axis: int, shift: int) -> ndarray:
    """
    Shift X along a spatial axis, padding boundary with zeros.
    Axis 0,1,2 correspond to width, height, depth (spatial dims 1,2,3 in X).

    :param X: (num_sample, W, H, D, num_label)
    :return:  (num_sample, W, H, D, num_label)  — neighbor values
    """
    spatial_axis = axis + 1  # offset by 1 due to batch dim
    result = np.roll(X, shift=-shift, axis=spatial_axis)

    # zero-out the wrapped boundary (treat boundary as "no neighbor")
    slices = [slice(None)] * X.ndim
    if shift > 0:
        slices[spatial_axis] = slice(-shift, None)   # last `shift` slices
    else:
        slices[spatial_axis] = slice(None, -shift)   # first `|shift|` slices
    result[tuple(slices)] = 0.0
    return result


def adjacency_cost(
    X: ndarray,
    C_adj: ndarray,
    directions: list[tuple[int, int]] = _DIRECTIONS,
) -> ndarray:
    """
    Compute total adjacency cost: Σ_{uv∈E} Σ_{l,l'} C[l,l'] * X[u,l] * X[v,l']

    Each directed edge (v → neighbor) is counted once per direction entry.
    Using all 6 directions double-counts undirected edges, so pass only
    3 positive-shift directions if you want exact undirected cost.

    :param X:     (num_sample, W, H, D, num_label)
    :param C_adj: (num_label, num_label) — shared across all edges
    :param directions: which neighbor directions to include
    :return:      (num_sample, 1)
    """
    assert X.ndim == 5, f"X must be 5D, got {X.ndim}"
    assert C_adj.ndim == 2 and C_adj.shape[0] == C_adj.shape[1] == X.shape[-1]

    total = np.zeros(X.shape[0], dtype=X.dtype)

    for axis, shift in directions:
        X_neighbor = _shift_X(X, axis, shift)  # (N, W, H, D, L)

        # For each voxel v: cost_v = Σ_{l,l'} C[l,l'] * X[v,l] * X_neighbor[v,l']
        # = X[v] @ C @ X_neighbor[v]^T  → scalar per voxel
        # Vectorized via einsum:
        #   X:          (N, W, H, D, l )
        #   X_neighbor: (N, W, H, D, l')
        #   C:          (l, l')
        # result:       (N,)
        cost = np.einsum('nwhdi,ij,nwhdj->n', X, C_adj, X_neighbor)
        total += cost

    return total.reshape(-1, 1)


# ─────────────────────────────────────────────
# Total Cost Builder
# ─────────────────────────────────────────────

def total_cost(
    C_unary: ndarray,
    X: ndarray,
    C_adj: ndarray,
    undirected: bool = True,
) -> ndarray:
    """
    Full objective: unary + adjacency terms.

    :param C_unary:    (N, W, H, D, L)
    :param X:          (N, W, H, D, L)
    :param C_adj:      (L, L)
    :param undirected: if True, use only 3 directions to avoid double-counting
    :return:           (N, 1)
    """
    directions = [(0,1),(1,1),(2,1)] if undirected else _DIRECTIONS
    return (
        labeling_cost(C_unary, X)
        + adjacency_cost(X, C_adj, directions=directions)
    )
