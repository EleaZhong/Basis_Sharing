"""
Baseline evaluation script for uncompressed models.

Computes perplexity on WikiText using lm_eval for the original model (no compression)
to provide a baseline for comparison with compressed models.
"""

from __future__ import annotations

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from config import ShareConfig, add_args


def main():
    """Main baseline evaluation function."""
    cmd_args = add_args()
    config = ShareConfig(cmd_args)

    print("=" * 60)
    print("BASELINE MODEL EVALUATION (No Compression)")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print("=" * 60)

    # Load tokenizer
    if config.model_type == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.pad_token or "[PAD]"

    # Load baseline model (no compression)
    print(f"\nLoading baseline model: {config.model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto")
    model.config.use_cache = False

    # Create HFLM wrapper for lm_eval
    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
    )

    # Compute perplexity using lm_eval
    print("\nEvaluating perplexity on WikiText...")
    results = lm_eval.simple_evaluate(
        model=hflm,
        tasks=["wikitext"],
        device=device,
        batch_size=config.calib_batch_size,
    )

    # Extract and print perplexity
    print("\n" + "=" * 60)
    print("Baseline Evaluation Results")
    print("=" * 60)

    word_perplexity = None
    if "results" in results:
        for task, metrics in results["results"].items():
            print(f"\n{task}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                    if "word_perplexity" in metric_name:
                        word_perplexity = value
                else:
                    print(f"  {metric_name}: {value}")

    print("=" * 60)

    return word_perplexity


if __name__ == "__main__":
    main()

