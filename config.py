"""
Configuration management for Basis Sharing.

This module provides configuration loading from YAML files and command-line arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def add_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Basis Sharing Model Compression")

    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="Path to YAML configuration file",
        type=str,
        default="",
    )
    parser.add_argument(
        "--calibration_size",
        "--cs",
        help="Number of samples for calibration",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dataset_name",
        help="Dataset name for calibration/evaluation",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_cache_dir",
        help="Directory to cache datasets",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--compression_ratio",
        help="Compression ratio as percentage (e.g., 20 for 20%%)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--group_size",
        help="Number of layers to group for basis sharing",
        type=int,
        default=None,
    )

    args, _ = parser.parse_known_args()
    return args


class ShareConfig:
    """Configuration class for Basis Sharing compression.

    Loads configuration from YAML files and command-line arguments.

    Attributes:
        model_type: Type of model (llama2, gpt2, opt, mistral)
        model_name: HuggingFace model name
        group_size: Number of layers per group
        compression_ratio: Target compression percentage
        context_length: Context length for evaluation
        stride: Stride for perplexity computation
        dataset_name: Dataset for calibration
        calibration_size: Number of calibration samples
        calib_batch_size: Batch size for calibration
    """

    # Map model names to short identifiers
    name_map: dict[str, str] = {
        "meta-llama/Llama-2-7b-hf": "llama2-7b",
        "jeffwan/llama-7b-hf": "llama2-7b",
        "jeffwan/llama-13b-hf": "llama2-13b",
        "jeffwan/llama-30b-hf": "llama2-30b",
        "gpt2": "gpt2",
        "facebook/opt-6.7b": "opt-6.7b",
        "mistralai/Mistral-7B-v0.1": "mistral-7b",
    }

    # Weight dimensions for different models: {proj_name: (in_features, out_features)}
    weight_info: dict[str, dict[str, tuple[int, int]]] = {
        "llama2-7b": {
            "self_attn.k_proj": (4096, 4096),
            "self_attn.q_proj": (4096, 4096),
            "self_attn.v_proj": (4096, 4096),
            "self_attn.o_proj": (4096, 4096),
            "mlp.up_proj": (4096, 11008),
            "mlp.gate_proj": (4096, 11008),
            "mlp.down_proj": (11008, 4096),
        },
        "llama2-13b": {
            "self_attn.k_proj": (5120, 5120),
            "self_attn.q_proj": (5120, 5120),
            "self_attn.v_proj": (5120, 5120),
            "self_attn.o_proj": (5120, 5120),
            "mlp.up_proj": (5120, 13824),
            "mlp.gate_proj": (5120, 13824),
            "mlp.down_proj": (13824, 5120),
        },
        "llama2-30b": {
            "self_attn.k_proj": (6656, 6656),
            "self_attn.q_proj": (6656, 6656),
            "self_attn.v_proj": (6656, 6656),
            "self_attn.o_proj": (6656, 6656),
            "mlp.up_proj": (6656, 17920),
            "mlp.gate_proj": (6656, 17920),
            "mlp.down_proj": (17920, 6656),
        },
        "gpt2": {
            "attn.c_attn": (768, 2304),
            "attn.c_proj": (768, 768),
            "mlp.c_fc": (768, 3072),
            "mlp.c_proj": (3072, 768),
        },
        "opt-6.7b": {
            "self_attn.k_proj": (4096, 4096),
            "self_attn.q_proj": (4096, 4096),
            "self_attn.v_proj": (4096, 4096),
            "self_attn.out_proj": (4096, 4096),
            "fc1": (4096, 16384),
            "fc2": (16384, 4096),
        },
        "mistral-7b": {
            "self_attn.k_proj": (4096, 1024),
            "self_attn.q_proj": (4096, 4096),
            "self_attn.v_proj": (4096, 1024),
            "self_attn.o_proj": (4096, 4096),
            "mlp.up_proj": (4096, 14336),
            "mlp.gate_proj": (4096, 14336),
            "mlp.down_proj": (14336, 4096),
        },
    }

    def __init__(self, cmd_args: argparse.Namespace):
        """Initialize configuration from command-line arguments.

        Args:
            cmd_args: Parsed command-line arguments
        """
        # Set defaults first
        self._set_defaults()

        # Load YAML config if provided
        if cmd_args.yaml_config_file:
            configuration = self._load_yaml_config(cmd_args.yaml_config_file)
            self._set_attr_from_config(configuration)

        # Override with command-line arguments (only if explicitly provided)
        cmd_args_dict = cmd_args.__dict__
        for arg_key, arg_val in cmd_args_dict.items():
            if arg_val is not None:
                setattr(self, arg_key, arg_val)

    def _set_defaults(self):
        """Set default values for all attributes."""
        self.model_type = "llama2"
        self.model_name = "meta-llama/Llama-2-7b-hf"
        self.group_size = 2
        self.compression_ratio = 20
        self.context_length = 2048
        self.stride = 2048
        self.dataset_name = "wikitext"
        self.calibration_size = 256
        self.calib_batch_size = 16
        self.dataset_cache_dir = None
        self.compressed_model_path = None

    @staticmethod
    def _load_yaml_config(yaml_path: str | Path) -> dict[str, Any]:
        """Load configuration from YAML file."""
        with open(yaml_path) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(f"YAML error in {yaml_path}: {exc}") from exc

    def _set_attr_from_config(self, configuration: dict[str, Any]):
        """Set attributes from configuration dictionary."""
        for _, param_family in configuration.items():
            if isinstance(param_family, dict):
                for key, val in param_family.items():
                    setattr(self, key, val)

    def get_target_patterns(self) -> list[str]:
        """Get regex patterns for target modules based on model type."""
        patterns = {
            "llama2": [
                r"self_attn\.k_proj",
                r"self_attn\.q_proj",
                r"self_attn\.v_proj",
                r"self_attn\.o_proj",
                r"mlp\.up_proj",
                r"mlp\.gate_proj",
                r"mlp\.down_proj",
            ],
            "mistral": [
                r"self_attn\.k_proj",
                r"self_attn\.q_proj",
                r"self_attn\.v_proj",
                r"self_attn\.o_proj",
                r"mlp\.up_proj",
                r"mlp\.gate_proj",
                r"mlp\.down_proj",
            ],
            "opt": [
                r"self_attn\.k_proj",
                r"self_attn\.q_proj",
                r"self_attn\.v_proj",
                r"self_attn\.out_proj",
                r"\.fc1$",
                r"\.fc2$",
            ],
            "gpt2": [
                r"attn\.c_attn",
                r"attn\.c_proj",
                r"mlp\.c_fc",
                r"mlp\.c_proj",
            ],
        }
        return patterns.get(self.model_type, patterns["llama2"])

    def to_basis_sharing_config(self):
        """Convert to BasisSharingConfig."""
        from basis_sharing import BasisSharingConfig

        return BasisSharingConfig(
            target_modules=self.get_target_patterns(),
            group_size=self.group_size,
            compression_ratio=self.compression_ratio / 100,
            exclude_layers=[],
            exclude_modules=[r"lm_head", r"embed_tokens", r"wte", r"wpe"],
        )

    def __repr__(self) -> str:
        """String representation of config."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"ShareConfig({attrs})"
