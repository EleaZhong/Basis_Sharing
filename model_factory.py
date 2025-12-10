"""
Model Factory for Basis Sharing Compression.

This module provides functions to create and compress models using the
model-agnostic BasisSharingWrapper.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from basis_sharing import BasisSharingConfig, BasisSharingWrapper
from config import ShareConfig
from prepare_data import prepare_data

if TYPE_CHECKING:
    pass


# Default target module patterns for common architectures
MODEL_TARGET_PATTERNS: dict[str, list[str]] = {
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

# Modules to exclude from compression
EXCLUDE_PATTERNS: list[str] = [
    r"lm_head",
    r"embed_tokens",
    r"wte",
    r"wpe",
    r"embed_positions",
    r"project_in",
    r"project_out",
]


def get_target_patterns(model_type: str) -> list[str]:
    """Get target module patterns for a model type."""
    return MODEL_TARGET_PATTERNS.get(model_type, MODEL_TARGET_PATTERNS["llama2"])


def get_tokenizer(config: ShareConfig):
    """Get the appropriate tokenizer for the model."""
    if config.model_type == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.pad_token or "[PAD]"
    return tokenizer


def create_compressed_model(
    config: ShareConfig,
    save_path: str | None = None,
) -> BasisSharingWrapper:
    """Create a compressed model using BasisSharingWrapper.

    This function:
    1. Loads the base model from HuggingFace
    2. Wraps it with BasisSharingWrapper
    3. Runs calibration
    4. Compresses the model
    5. Optionally saves the compressed model

    Args:
        config: ShareConfig with model and compression settings
        save_path: Optional path to save the compressed model

    Returns:
        BasisSharingWrapper containing the compressed model
    """
    # Determine save path from config if not provided
    if save_path is None:
        save_path = getattr(config, "compressed_model_path", None)

    # Check for existing compressed model
    if save_path and os.path.exists(save_path):
        print(f"Loading existing compressed model from {save_path}")
        return load_compressed_model(config, save_path)

    # Load tokenizer
    tokenizer = get_tokenizer(config)

    # Load base model
    print(f"Loading base model: {config.model_name}")
    model_kwargs: dict = {"device_map": "auto"}

    if hasattr(config, "model_name") and "30b" in config.model_name:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.config.use_cache = False

    # Get target patterns for this model type
    target_patterns = get_target_patterns(config.model_type)

    # Create BasisSharingConfig
    basis_config = BasisSharingConfig(
        target_modules=target_patterns,
        group_size=config.group_size,
        compression_ratio=config.compression_ratio / 100,  # Convert from percentage
        exclude_modules=EXCLUDE_PATTERNS,
        exclude_layers=[],
    )

    # Create wrapper
    print("Creating BasisSharingWrapper...")
    wrapper = BasisSharingWrapper(model, basis_config)

    # Prepare calibration data
    print("Preparing calibration data...")
    train_dataset, _, _, data_collator = prepare_data(
        config.dataset_name,
        tokenizer,
        config.context_length,
        getattr(config, "dataset_cache_dir", None),
    )

    # Create calibration dataloader
    torch.manual_seed(2023)
    calib_size = getattr(config, "calibration_size", 256)
    indices = torch.randperm(len(train_dataset))[:calib_size]
    subset = Subset(train_dataset, indices.tolist())

    batch_size = getattr(config, "calib_batch_size", 16)
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4,
    )

    # Run calibration
    print("Running calibration...")
    wrapper.calibrate(dataloader)

    # Compress
    print("Compressing model...")
    wrapper.compress()

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        wrapper.save_compressed(save_path)
        tokenizer.save_pretrained(os.path.dirname(save_path))

    return wrapper


def load_compressed_model(
    config: ShareConfig,
    path: str,
    device: str | torch.device = "auto",
) -> BasisSharingWrapper:
    """Load a previously compressed model.

    Args:
        config: ShareConfig with model settings
        path: Path to saved compressed model
        device: Device to load model on

    Returns:
        BasisSharingWrapper with loaded compressed model
    """
    # Load base model structure
    model_kwargs: dict = {}
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": device}

    if hasattr(config, "model_name") and "30b" in config.model_name:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    # Load compressed state
    wrapper = BasisSharingWrapper.load_compressed(model, path, device=device)

    return wrapper
