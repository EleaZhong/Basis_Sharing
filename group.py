"""
Grouping utilities for Basis Sharing.

This module provides utility functions for creating layer groups and
computing SVD for basis sharing compression.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def create_layer_groups(
    num_layers: int,
    group_size: int,
    exclude_layers: Sequence[int] | None = None,
) -> list[list[int]]:
    """Create layer groups for basis sharing.

    Args:
        num_layers: Total number of layers in the model
        group_size: Number of layers per group
        exclude_layers: Layer indices to exclude from grouping

    Returns:
        List of groups, where each group is a list of layer indices
    """
    exclude_set = set(exclude_layers or [])
    layer_indices = [i for i in range(num_layers) if i not in exclude_set]

    groups = []
    for i in range(0, len(layer_indices), group_size):
        group = layer_indices[i : i + group_size]
        if group:  # Don't add empty groups
            groups.append(group)

    return groups


def compute_num_basis(
    in_features: int,
    out_features: int,
    group_size: int,
    compression_ratio: float,
) -> int:
    """Compute the number of basis vectors for target compression ratio.

    The compression ratio is defined as:
        compressed_size / original_size = compression_ratio

    Where:
        original_size = in_features * out_features * group_size
        compressed_size = in_features * num_basis + num_basis * out_features * group_size

    Args:
        in_features: Input dimension of the linear layer
        out_features: Output dimension of the linear layer
        group_size: Number of layers sharing the basis
        compression_ratio: Target ratio (0.2 means 20% of original size)

    Returns:
        Number of basis vectors to use
    """
    # Ensure compression_ratio is a fraction, not percentage
    if compression_ratio > 1:
        compression_ratio = compression_ratio / 100

    total_original = in_features * out_features * group_size

    # Solve for num_basis:
    # compressed = in_features * num_basis + num_basis * out_features * group_size
    # compressed = num_basis * (in_features + out_features * group_size)
    # num_basis = compressed / (in_features + out_features * group_size)
    # num_basis = (original * compression_ratio) / (in_features + out_features * group_size)

    num_basis = int(
        (total_original * compression_ratio) / (in_features + out_features * group_size)
    )

    return max(1, num_basis)


def run_svd_with_whitening(
    weights: list[torch.Tensor],
    s: torch.Tensor,
    s_inv: torch.Tensor,
    num_basis: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run whitened SVD on concatenated weight matrices.

    This applies the whitening transformation before SVD to account for
    input activation statistics (following SVD-LLM approach).

    Args:
        weights: List of weight matrices, each shape (in_features, out_features)
        s: Whitening matrix
        s_inv: Inverse whitening matrix
        num_basis: Number of basis vectors to keep

    Returns:
        Tuple of:
            - basis: Shared basis matrix (in_features, num_basis)
            - coefficients: List of per-layer coefficient matrices (num_basis, out_features)
    """
    # Concatenate weights: (in_features, out_features * num_layers)
    w_concat = torch.cat(weights, dim=-1).double()

    # Move to same device
    device = s.device
    w_concat = w_concat.to(device)
    s = s.to(device)
    s_inv = s_inv.to(device)

    # Apply whitening: W' = S @ W
    w_whitened = s @ w_concat

    # SVD: W' = U @ Sigma @ V^T
    u, sigma, v = torch.linalg.svd(w_whitened, full_matrices=False)

    # Truncate to num_basis
    u = u[:, :num_basis]
    sigma = sigma[:num_basis]
    v = v[:num_basis, :]

    # Compute basis: B = S^{-1} @ U @ diag(sigma)
    # Shape: (in_features, num_basis)
    basis = (s_inv @ u @ torch.diag(sigma)).float()

    # Split V^T into per-layer coefficients
    out_features = weights[0].shape[1]
    coefficients = []
    for i in range(len(weights)):
        start = i * out_features
        end = start + out_features
        coef = v[:, start:end].float()  # (num_basis, out_features)
        coefficients.append(coef)

    return basis, coefficients
