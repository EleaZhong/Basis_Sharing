"""
Evaluation script for Basis Sharing compressed models.

Computes perplexity on test datasets to evaluate model quality after compression.
"""

from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizer

from basis_sharing import BasisSharingWrapper
from config import ShareConfig, add_args
from model_factory import create_compressed_model
from prepare_data import prepare_data


def compute_ppl(
    max_length: int,
    stride: int,
    data,
    model: BasisSharingWrapper,
    device: str | torch.device,
) -> torch.Tensor:
    """Compute perplexity on a dataset.

    Args:
        max_length: Maximum sequence length
        stride: Stride for sliding window
        data: Dataset with input_ids
        model: BasisSharingWrapper to evaluate
        device: Device to run evaluation on

    Returns:
        Perplexity score
    """
    eval_model = model.model if isinstance(model, BasisSharingWrapper) else model
    eval_model.to(device)
    eval_model.eval()

    seq_len = data.input_ids.size(1)
    nlls = []
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="Computing PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = data.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            output = eval_model(input_ids, labels=target_ids)
            neg_log_likelihood = output.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl


def main():
    """Main evaluation function."""
    cmd_args = add_args()
    config = ShareConfig(cmd_args)

    print(f"Model: {config.model_name}")
    print(f"Compression ratio: {config.compression_ratio}%")
    print(f"Group size: {config.group_size}")

    # Load tokenizer
    if config.model_type == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.pad_token or "[PAD]"

    # Load datasets
    _, _, test_dataset, _ = prepare_data(
        config.dataset_name,
        tokenizer,
        config.context_length,
        getattr(config, "dataset_cache_dir", None),
    )

    # Create compressed model
    wrapper = create_compressed_model(config)

    # Compute perplexity
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppl = compute_ppl(config.context_length, config.stride, test_dataset, wrapper, device)

    print(f"\nPerplexity: {ppl.item():.2f}")


if __name__ == "__main__":
    main()
