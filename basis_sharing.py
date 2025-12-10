"""
Model-agnostic Basis Sharing Compression for Transformer Models.

This module provides a wrapper-based approach to compress any transformer model
by sharing basis matrices across grouped layers and using per-layer coefficients.

Key components:
- BasisSharingConfig: Configuration for compression parameters
- ShareLinear: Drop-in replacement for nn.Linear with basis sharing support
- BasisSharingWrapper: Wraps any model and manages the compression pipeline

Usage:
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    config = BasisSharingConfig(
        target_modules=["self_attn.*_proj", "mlp.*_proj"],
        group_size=2,
        compression_ratio=0.2,
    )
    wrapper = BasisSharingWrapper(model, config)
    wrapper.calibrate(dataloader)
    wrapper.compress()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterator
    from torch.utils.data import DataLoader


class ShareLinearState(Enum):
    """State machine for ShareLinear lifecycle."""

    ORIGINAL = auto()  # Forward uses original weights
    CALIBRATING = auto()  # Forward uses original weights + tracks inputs
    COMPRESSED = auto()  # Forward uses basis @ coefficient


@dataclass
class BasisSharingConfig:
    """Configuration for basis sharing compression.

    Attributes:
        target_modules: Regex patterns for modules to compress (e.g., ["self_attn.*_proj"])
        group_size: Number of layers to group together for basis sharing
        compression_ratio: Target compression ratio (0.2 = 20% of original size)
        exclude_layers: Layer indices to exclude from compression (e.g., [0, 31])
        exclude_modules: Regex patterns for modules to explicitly exclude
        share_qkv_basis: Whether Q, K, V projections share calibration data (same input)
        share_gate_up_basis: Whether gate and up projections share calibration data
    """

    target_modules: list[str]
    group_size: int = 2
    compression_ratio: float = 0.2
    exclude_layers: list[int] = field(default_factory=list)
    exclude_modules: list[str] = field(default_factory=list)
    share_qkv_basis: bool = True
    share_gate_up_basis: bool = True

    def should_target(self, name: str) -> bool:
        """Check if a module name should be targeted for compression."""
        # Check exclusions first
        for pattern in self.exclude_modules:
            if re.search(pattern, name):
                return False

        # Check if matches any target pattern
        for pattern in self.target_modules:
            if re.search(pattern, name):
                return True
        return False


class Basis(nn.Module):
    """Shared basis module (input_dim -> num_basis).

    This is the weight-tied component shared across grouped layers.
    """

    def __init__(self, in_features: int, num_basis: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.num_basis = num_basis
        self.weight = nn.Parameter(
            torch.empty(num_basis, in_features, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def set_weight(self, weight: torch.Tensor):
        """Set weight from SVD result (shape: in_features x num_basis)."""
        with torch.no_grad():
            self.weight.copy_(weight.T.detach().clone())


class Coefficient(nn.Module):
    """Per-layer coefficient module (num_basis -> out_features).

    Each layer in a group has its own coefficient while sharing the basis.
    """

    def __init__(
        self,
        num_basis: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_basis = num_basis
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, num_basis, device=device, dtype=dtype)
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
            if bias
            else None
        )
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def set_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        """Set weight from SVD result (shape: num_basis x out_features)."""
        with torch.no_grad():
            self.weight.copy_(weight.T.detach().clone())
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias.detach().clone())


class ShareLinear(nn.Module):
    """Drop-in replacement for nn.Linear with basis sharing support.

    Lifecycle:
    1. ORIGINAL: Created from existing nn.Linear, uses original weights
    2. CALIBRATING: Same as ORIGINAL but tracks input statistics (X^T @ X)
    3. COMPRESSED: Uses basis (shared) @ coefficient (per-layer)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self._state = ShareLinearState.ORIGINAL

        # Original weights (temporary, cleared after compression)
        self.original_weight: nn.Parameter | None = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.original_bias: nn.Parameter | None = (
            nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
            if bias
            else None
        )

        # Calibration accumulator (X^T @ X)
        self.register_buffer("calib_xtx", None)
        self.calib_count: int = 0

        # Compressed components (set after SVD)
        self.basis: Basis | None = None
        self.coefficient: Coefficient | None = None

        # Metadata
        self.name: str = ""
        self.layer_idx: int = -1
        self.proj_type: str = ""  # e.g., "k_proj", "q_proj", "up_proj"

    @property
    def state(self) -> ShareLinearState:
        return self._state

    @state.setter
    def state(self, value: ShareLinearState):
        self._state = value

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        name: str = "",
        layer_idx: int = -1,
        proj_type: str = "",
    ) -> ShareLinear:
        """Create ShareLinear from existing nn.Linear, copying weights."""
        share = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        share.original_weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            share.original_bias = nn.Parameter(linear.bias.data.clone())
        share.name = name
        share.layer_idx = layer_idx
        share.proj_type = proj_type
        return share

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._state == ShareLinearState.COMPRESSED:
            # Compressed forward: basis -> coefficient
            if self.basis is None or self.coefficient is None:
                raise RuntimeError(
                    f"ShareLinear '{self.name}' is in COMPRESSED state but "
                    "basis/coefficient not set"
                )
            out = self.coefficient(self.basis(x))
            # Add original bias if it was preserved
            if self.original_bias is not None:
                out = out + self.original_bias
            return out

        # Original/Calibrating forward
        if self._state == ShareLinearState.CALIBRATING:
            self._track_input(x)

        return F.linear(x, self.original_weight, self.original_bias)

    @torch.no_grad()
    def _track_input(self, x: torch.Tensor):
        """Accumulate X^T @ X for whitening matrix computation.
        
        Keeps accumulation on GPU to avoid slow CPU transfers during calibration.
        Data is moved to CPU only when retrieved via get_calibration_data().
        """
        inp = x.detach().float()
        inp = inp.flatten(start_dim=0, end_dim=-2)  # (batch*seq, hidden)
        xtx = inp.T @ inp  # Keep on same device as input

        if self.calib_xtx is None:
            self.calib_xtx = xtx
        else:
            # Handle potential device mismatch in edge cases
            if self.calib_xtx.device != xtx.device:
                self.calib_xtx = self.calib_xtx.to(xtx.device)
            self.calib_xtx = self.calib_xtx + xtx
        self.calib_count += 1

    def get_calibration_data(self) -> torch.Tensor | None:
        """Return accumulated calibration data, moved to CPU for SVD computation."""
        if self.calib_xtx is None:
            return None
        return self.calib_xtx.cpu()

    def clear_calibration_data(self):
        """Clear calibration data to free memory."""
        self.calib_xtx = None
        self.calib_count = 0

    def get_weight_for_svd(self) -> torch.Tensor:
        """Get weight matrix in correct orientation for SVD.

        Returns weight as (in_features, out_features) for SVD computation.
        """
        if self.original_weight is None:
            raise RuntimeError("Original weight already cleared")
        # Standard nn.Linear stores as (out_features, in_features)
        # We need (in_features, out_features) for SVD
        return self.original_weight.data.T

    def setup_compressed(
        self, basis: Basis, coefficient: Coefficient, keep_bias: bool = True
    ):
        """Setup compressed mode with shared basis and per-layer coefficient."""
        self.basis = basis
        self.coefficient = coefficient

        # Clear original weight to free memory
        self.original_weight = None
        if not keep_bias:
            self.original_bias = None

        self._state = ShareLinearState.COMPRESSED

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.has_bias}, state={self._state.name}"
        )


class BasisSharingWrapper(nn.Module):
    """Model-agnostic wrapper for basis sharing compression.

    This wrapper:
    1. Replaces target nn.Linear modules with ShareLinear
    2. Manages calibration (input statistics collection)
    3. Computes SVD and sets up weight-tied basis matrices
    4. Provides seamless forward pass delegation

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> config = BasisSharingConfig(
        ...     target_modules=["self_attn.*_proj", "mlp.*_proj"],
        ...     group_size=2,
        ...     compression_ratio=0.2,
        ... )
        >>> wrapper = BasisSharingWrapper(model, config)
        >>> wrapper.calibrate(dataloader)
        >>> wrapper.compress()
        >>> # Now use wrapper for inference
        >>> outputs = wrapper(input_ids=input_ids)
    """

    # Known projection type patterns for different model architectures
    PROJ_PATTERNS = {
        # LLaMA/Mistral style
        "k_proj": r"self_attn\.k_proj",
        "q_proj": r"self_attn\.q_proj",
        "v_proj": r"self_attn\.v_proj",
        "o_proj": r"self_attn\.o_proj",
        "up_proj": r"mlp\.up_proj",
        "gate_proj": r"mlp\.gate_proj",
        "down_proj": r"mlp\.down_proj",
        # OPT style
        "k_proj_opt": r"self_attn\.k_proj",
        "q_proj_opt": r"self_attn\.q_proj",
        "v_proj_opt": r"self_attn\.v_proj",
        "out_proj": r"self_attn\.out_proj",
        "fc1": r"\.fc1$",
        "fc2": r"\.fc2$",
        # GPT-2 style (fused QKV)
        "c_attn": r"attn\.c_attn",
        "c_proj": r"attn\.c_proj",
        "c_fc": r"mlp\.c_fc",
        "mlp_c_proj": r"mlp\.c_proj",
    }

    def __init__(self, model: nn.Module, config: BasisSharingConfig):
        super().__init__()
        self.model = model
        self.config = config

        # Track all ShareLinear modules by name
        self.share_linears: dict[str, ShareLinear] = {}

        # Track ShareLinears grouped by projection type and layer
        # Structure: {proj_type: {layer_idx: ShareLinear}}
        self.proj_layers: dict[str, dict[int, ShareLinear]] = {}

        # Shared basis modules (for weight tying and serialization)
        # Structure: {proj_type: {group_leader_idx: Basis}}
        self.shared_bases: nn.ModuleDict = nn.ModuleDict()

        # Compression metadata
        self.groups: dict[str, list[list[int]]] = {}
        self.num_basis: dict[str, int] = {}
        self._is_compressed = False

        # Detect model architecture
        self._num_layers = self._detect_num_layers()

        # Replace target modules
        self._replace_target_linears()

    def _detect_num_layers(self) -> int:
        """Detect number of transformer layers in the model."""
        # Try common patterns
        for name, module in self.model.named_modules():
            if hasattr(module, "__len__") and any(
                pattern in name
                for pattern in ["layers", "h", "decoder.layers", "encoder.layers"]
            ):
                if isinstance(module, nn.ModuleList):
                    return len(module)

        # Fallback: count unique layer indices from module names
        layer_indices = set()
        for name, _ in self.model.named_modules():
            match = re.search(r"\.(\d+)\.", name)
            if match:
                layer_indices.add(int(match.group(1)))
        return len(layer_indices) if layer_indices else 0

    def _extract_layer_idx(self, name: str) -> int:
        """Extract layer index from module name."""
        # Match patterns like "layers.0.", "h.0.", "decoder.layers.0."
        match = re.search(r"\.(\d+)\.", name)
        if match:
            return int(match.group(1))
        return -1

    def _extract_proj_type(self, name: str) -> str:
        """Extract projection type from module name."""
        for proj_type, pattern in self.PROJ_PATTERNS.items():
            if re.search(pattern, name):
                return proj_type

        # Fallback: use last component of name
        parts = name.split(".")
        return parts[-1] if parts else "unknown"

    def _should_replace(self, name: str, module: nn.Module) -> bool:
        """Check if module should be replaced with ShareLinear."""
        if not isinstance(module, nn.Linear):
            return False

        # Check config patterns
        if not self.config.should_target(name):
            return False

        # Check layer exclusions
        layer_idx = self._extract_layer_idx(name)
        if layer_idx in self.config.exclude_layers:
            return False

        return True

    def _set_module_by_name(self, name: str, new_module: nn.Module):
        """Replace a module in the model by its name."""
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def _replace_target_linears(self):
        """Replace all target nn.Linear modules with ShareLinear."""
        modules_to_replace = []

        for name, module in self.model.named_modules():
            if self._should_replace(name, module):
                modules_to_replace.append((name, module))

        for name, module in modules_to_replace:
            layer_idx = self._extract_layer_idx(name)
            proj_type = self._extract_proj_type(name)

            share_linear = ShareLinear.from_linear(
                module, name=name, layer_idx=layer_idx, proj_type=proj_type
            )

            self._set_module_by_name(name, share_linear)
            self.share_linears[name] = share_linear

            # Organize by projection type
            if proj_type not in self.proj_layers:
                self.proj_layers[proj_type] = {}
            self.proj_layers[proj_type][layer_idx] = share_linear

        print(f"Replaced {len(self.share_linears)} Linear modules with ShareLinear")
        for proj_type, layers in self.proj_layers.items():
            print(f"  {proj_type}: {len(layers)} layers")

    def start_calibration(self):
        """Start calibration mode - ShareLinears will track input statistics."""
        for sl in self.share_linears.values():
            sl.state = ShareLinearState.CALIBRATING
            sl.clear_calibration_data()

    def end_calibration(self):
        """End calibration mode."""
        for sl in self.share_linears.values():
            sl.state = ShareLinearState.ORIGINAL

    @torch.no_grad()
    def calibrate(
        self,
        dataloader: DataLoader,
        max_samples: int | None = None,
        device: str | torch.device | None = None,
    ):
        """Run calibration to collect input statistics.

        Args:
            dataloader: DataLoader providing calibration samples
            max_samples: Maximum number of samples to use (None = use all)
            device: Device to run calibration on
        """
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        self.start_calibration()

        # Disable caching for calibration
        use_cache = getattr(self.model.config, "use_cache", None)
        if use_cache is not None:
            self.model.config.use_cache = False

        try:
            num_samples = 0
            for batch in tqdm(dataloader, desc="Calibrating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                _ = self.model(**batch)

                num_samples += 1
                if max_samples is not None and num_samples >= max_samples:
                    break

            print(f"Calibration complete: {num_samples} batches processed")

        finally:
            self.end_calibration()
            if use_cache is not None:
                self.model.config.use_cache = use_cache

    def _get_calibration_data_for_proj(
        self, proj_type: str, layer_indices: list[int]
    ) -> torch.Tensor:
        """Get aggregated calibration data for a projection type across layers.

        For projections that share input (Q/K/V share attention input, gate/up share MLP input),
        we can use calibration data from any one of them.
        """
        # Handle shared calibration data
        if self.config.share_qkv_basis:
            if proj_type in ("q_proj", "v_proj"):
                proj_type = "k_proj"
            elif proj_type in ("q_proj_opt", "v_proj_opt"):
                proj_type = "k_proj_opt"

        if self.config.share_gate_up_basis:
            if proj_type == "gate_proj":
                proj_type = "up_proj"

        layers = self.proj_layers.get(proj_type, {})
        data = None

        for idx in layer_indices:
            if idx not in layers:
                continue
            layer_data = layers[idx].get_calibration_data()
            if layer_data is not None:
                if data is None:
                    data = layer_data.clone()
                else:
                    data = data + layer_data

        if data is None:
            raise RuntimeError(
                f"No calibration data for {proj_type} layers {layer_indices}"
            )

        return data

    def _compute_whitening(
        self, calib_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute whitening matrix S and its inverse from calibration data.

        Uses Cholesky decomposition following SVD-LLM approach.

        Returns:
            (S, S_inv): Whitening matrix and its inverse
        """
        data = calib_data.double()

        try:
            scaling_diag_matrix = torch.linalg.cholesky(data).T
        except RuntimeError:
            # Handle non-positive definite matrix
            print("Warning: Calibration matrix not positive definite, adding regularization")
            eigenvalues = torch.linalg.eigvalsh(data)
            data = data + (-eigenvalues[0] + 1e-5) * torch.eye(
                data.shape[0], device=data.device, dtype=data.dtype
            )
            scaling_diag_matrix = torch.linalg.cholesky(data).T

        s_inv = torch.linalg.inv(scaling_diag_matrix)
        return scaling_diag_matrix, s_inv

    def _compute_num_basis(
        self, in_features: int, out_features: int, group_size: int
    ) -> int:
        """Compute number of basis vectors for target compression ratio."""
        compression_ratio = 1 - self.config.compression_ratio
        total_original = in_features * out_features * group_size
        # Compressed size: in_features * num_basis + num_basis * out_features * group_size
        # Solving for num_basis given compression ratio
        num_basis = int(
            (total_original * compression_ratio) / (in_features + out_features * group_size)
        )
        return max(1, num_basis)

    def _create_groups(self) -> dict[str, list[list[int]]]:
        """Create layer groups based on config.group_size."""
        groups = {}

        for proj_type, layers in self.proj_layers.items():
            layer_indices = sorted(layers.keys())

            # Filter excluded layers
            layer_indices = [
                idx for idx in layer_indices if idx not in self.config.exclude_layers
            ]

            # Create groups
            proj_groups = []
            for i in range(0, len(layer_indices), self.config.group_size):
                group = layer_indices[i : i + self.config.group_size]
                if group:  # Don't add empty groups
                    proj_groups.append(group)

            groups[proj_type] = proj_groups

        return groups

    def _run_svd_for_group(
        self,
        proj_type: str,
        group: list[int],
        s: torch.Tensor,
        s_inv: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run SVD for a group of layers.

        Args:
            proj_type: Type of projection (e.g., "k_proj")
            group: List of layer indices in this group
            s: Whitening matrix
            s_inv: Inverse whitening matrix

        Returns:
            (basis, coefficients): Shared basis and list of per-layer coefficients
        """
        layers = self.proj_layers[proj_type]

        # Collect weights from all layers in group
        weights = []
        for layer_idx in group:
            w = layers[layer_idx].get_weight_for_svd()  # (in, out)
            weights.append(w)

        # Concatenate: (in_features, out_features * group_size)
        w_concat = torch.cat(weights, dim=-1).double()

        # Move to same device as whitening matrices
        device = s.device
        w_concat = w_concat.to(device)
        s = s.to(device)
        s_inv = s_inv.to(device)

        # Apply whitening: W' = S @ W
        w_whitened = s @ w_concat

        # SVD: W' = U @ Sigma @ V^T
        u, sigma, v = torch.linalg.svd(w_whitened, full_matrices=False)

        # Compute number of basis vectors
        in_features = layers[group[0]].in_features
        out_features = layers[group[0]].out_features
        num_basis = self._compute_num_basis(in_features, out_features, len(group))

        # Truncate to num_basis
        u = u[:, :num_basis]
        sigma = sigma[:num_basis]
        v = v[:num_basis, :]

        # Basis = S^{-1} @ U @ diag(sigma)
        # Shape: (in_features, num_basis)
        basis = (s_inv @ u @ torch.diag(sigma)).float()

        # Split V^T into per-layer coefficients
        # V^T shape: (num_basis, out_features * group_size)
        coefficients = []
        for i in range(len(group)):
            start = i * out_features
            end = start + out_features
            coef = v[:, start:end].float()  # (num_basis, out_features)
            coefficients.append(coef)

        # Store num_basis for this projection type
        self.num_basis[proj_type] = num_basis

        return basis, coefficients

    def compress(self):
        """Run compression: compute SVD and setup weight-tied basis matrices.

        Must be called after calibrate().
        """
        if self._is_compressed:
            print("Model already compressed")
            return

        # Create groups
        self.groups = self._create_groups()
        print(f"Created groups for {len(self.groups)} projection types")

        for proj_type, proj_groups in tqdm(
            self.groups.items(), desc="Compressing projections"
        ):
            if not proj_groups:
                continue

            # Initialize ModuleDict for this projection type's shared bases
            self.shared_bases[proj_type] = nn.ModuleDict()

            layers = self.proj_layers[proj_type]

            for group in proj_groups:
                # Get calibration data for this group
                calib_data = self._get_calibration_data_for_proj(proj_type, group)

                # Compute whitening matrix
                s, s_inv = self._compute_whitening(calib_data)

                # Run SVD
                basis_weights, coefficients = self._run_svd_for_group(
                    proj_type, group, s, s_inv
                )

                # Create shared basis module
                in_features = layers[group[0]].in_features
                num_basis = basis_weights.shape[1]

                shared_basis = Basis(
                    in_features,
                    num_basis,
                    device=basis_weights.device,
                    dtype=basis_weights.dtype,
                )
                shared_basis.set_weight(basis_weights)

                # Store shared basis (use first layer index as key)
                group_key = str(group[0])
                self.shared_bases[proj_type][group_key] = shared_basis

                # Setup each layer with shared basis and its coefficient
                for i, layer_idx in enumerate(group):
                    share_linear = layers[layer_idx]
                    out_features = share_linear.out_features

                    # Create coefficient for this layer
                    coefficient = Coefficient(
                        num_basis,
                        out_features,
                        bias=False,  # Original bias is handled separately
                        device=coefficients[i].device,
                        dtype=coefficients[i].dtype,
                    )
                    coefficient.set_weight(coefficients[i])

                    # Setup compressed mode
                    share_linear.setup_compressed(
                        basis=shared_basis,
                        coefficient=coefficient,
                        keep_bias=True,
                    )

        # Clear calibration data
        for sl in self.share_linears.values():
            sl.clear_calibration_data()

        self._is_compressed = True
        print("Compression complete!")
        self._print_compression_stats()

    def _print_compression_stats(self):
        """Print compression statistics."""
        total_original = 0
        total_compressed = 0

        for proj_type, layers in self.proj_layers.items():
            if proj_type not in self.num_basis:
                continue

            num_basis = self.num_basis[proj_type]

            for layer_idx, share_linear in layers.items():
                in_f = share_linear.in_features
                out_f = share_linear.out_features

                # Original: in_features * out_features
                original = in_f * out_f
                total_original += original

        # Count compressed parameters
        for proj_type, bases_dict in self.shared_bases.items():
            for group_key, basis in bases_dict.items():
                # Basis parameters
                total_compressed += basis.weight.numel()

        for sl in self.share_linears.values():
            if sl.coefficient is not None:
                total_compressed += sl.coefficient.weight.numel()
            if sl.original_bias is not None:
                total_compressed += sl.original_bias.numel()

        ratio = total_compressed / total_original if total_original > 0 else 0
        print(f"Original parameters: {total_original:,}")
        print(f"Compressed parameters: {total_compressed:,}")
        print(f"Compression ratio: {ratio:.2%}")

    def forward(self, *args, **kwargs):
        """Forward pass delegated to wrapped model."""
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def save_compressed(self, path: str):
        """Save compressed model."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "shared_bases": self.shared_bases.state_dict(),
                "config": self.config,
                "groups": self.groups,
                "num_basis": self.num_basis,
            },
            path,
        )
        print(f"Saved compressed model to {path}")

    @classmethod
    def load_compressed(
        cls, model: nn.Module, path: str, device: str | torch.device = "cpu"
    ) -> BasisSharingWrapper:
        """Load a compressed model."""
        checkpoint = torch.load(path, map_location=device)

        wrapper = cls(model, checkpoint["config"])
        wrapper.groups = checkpoint["groups"]
        wrapper.num_basis = checkpoint["num_basis"]

        # Load model state dict
        wrapper.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # Load shared bases
        wrapper.shared_bases.load_state_dict(checkpoint["shared_bases"])

        wrapper._is_compressed = True
        return wrapper

    def get_model_for_training(self) -> nn.Module:
        """Get the underlying model for training (e.g., with LoRA)."""
        return self.model


# Convenience function for quick compression
def compress_model(
    model: nn.Module,
    dataloader: DataLoader,
    target_modules: list[str],
    group_size: int = 2,
    compression_ratio: float = 0.2,
    max_calibration_samples: int | None = None,
    exclude_layers: list[int] | None = None,
    device: str | torch.device | None = None,
) -> BasisSharingWrapper:
    """Convenience function to compress a model in one call.

    Args:
        model: The model to compress
        dataloader: DataLoader for calibration
        target_modules: Regex patterns for modules to compress
        group_size: Number of layers per group
        compression_ratio: Target compression ratio
        max_calibration_samples: Max calibration batches (None = all)
        exclude_layers: Layer indices to exclude
        device: Device for calibration

    Returns:
        BasisSharingWrapper with compressed model
    """
    config = BasisSharingConfig(
        target_modules=target_modules,
        group_size=group_size,
        compression_ratio=compression_ratio,
        exclude_layers=exclude_layers or [],
    )

    wrapper = BasisSharingWrapper(model, config)
    wrapper.calibrate(dataloader, max_samples=max_calibration_samples, device=device)
    wrapper.compress()

    return wrapper

