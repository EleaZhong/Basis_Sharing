"""
Calibration utilities for Basis Sharing.

This module provides utility functions for calibration data management
and whitening matrix computation.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import torch


def save_calibration_data(path: str | Path, data: torch.Tensor) -> None:
    """Save calibration data to disk.

    Args:
        path: File path to save to
        data: Calibration tensor (typically X^T @ X)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_calibration_data(path: str | Path) -> torch.Tensor:
    """Load calibration data from disk.

    Args:
        path: File path to load from

    Returns:
        Loaded calibration tensor
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_whitening_matrix(
    calib_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute whitening matrix S and its inverse from calibration data.

    Uses Cholesky decomposition following the SVD-LLM approach for
    numerical stability.

    Args:
        calib_data: Calibration data matrix (X^T @ X)

    Returns:
        Tuple of (S, S_inv) where:
            - S: Whitening matrix
            - S_inv: Inverse of whitening matrix
    """
    data = calib_data.double()

    try:
        # Cholesky decomposition: X^T @ X = L @ L^T
        # We use the transpose as the whitening matrix
        scaling_diag_matrix = torch.linalg.cholesky(data).T
    except RuntimeError:
        # Handle non-positive definite matrix by adding regularization
        print("Warning: Calibration matrix not positive definite, adding regularization")
        eigenvalues = torch.linalg.eigvalsh(data)
        # Add small positive value to make positive definite
        regularization = -eigenvalues[0] + 1e-5
        data = data + regularization * torch.eye(
            data.shape[0], device=data.device, dtype=data.dtype
        )
        scaling_diag_matrix = torch.linalg.cholesky(data).T

    # Compute inverse
    s_inv = torch.linalg.inv(scaling_diag_matrix)

    return scaling_diag_matrix, s_inv


def aggregate_calibration_data(data_list: list[torch.Tensor]) -> torch.Tensor:
    """Aggregate multiple calibration data tensors.

    Args:
        data_list: List of calibration tensors to aggregate

    Returns:
        Aggregated calibration data (sum of all inputs)
    """
    if not data_list:
        raise ValueError("Cannot aggregate empty list of calibration data")

    result = data_list[0].clone()
    for data in data_list[1:]:
        result = result + data

    return result
