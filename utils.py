"""
Utility functions for Basis Sharing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_num_basis(
    nx: int,
    nf: int,
    group_size: int,
    compression_ratio: float,
) -> int:
    """Compute the number of basis vectors for target compression ratio.

    Args:
        nx: Input dimension (hidden size)
        nf: Output dimension (typically projection size)
        group_size: Number of layers sharing the basis
        compression_ratio: Target compression percentage (e.g., 20 for 20%)

    Returns:
        Number of basis vectors to use
    """
    # Convert from percentage to fraction (e.g., 20 -> 0.8 for 80% reduction)
    compression_ratio = 1 - compression_ratio / 100

    # Original parameters: nx * nf * group_size
    total = nx * nf * group_size

    # Compressed parameters: nx * num_basis + num_basis * nf * group_size
    num_basis = (total * compression_ratio) // (nx + nf * group_size)

    return max(1, int(num_basis))


def get_device(model: nn.Module) -> str:
    """Get the device of a model."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_size(model: nn.Module, name: str = "Model") -> None:
    """Print model size information."""
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"{name} Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")

    # Estimate size in MB (assuming float32)
    size_mb = total_params * 4 / (1024 * 1024)
    print(f"  Estimated size (FP32): {size_mb:.2f} MB")


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True
