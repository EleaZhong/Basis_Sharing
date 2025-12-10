"""
Benchmark script for evaluating compressed models on standard NLP tasks.

Uses lm_eval library for evaluation on tasks like ARC, HellaSwag, PIQA, etc.
"""

from __future__ import annotations

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer, LlamaTokenizer

from config import ShareConfig, add_args
from model_factory import create_compressed_model


BENCHMARK_TASKS = [
    "wikitext",
    "openbookqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "hellaswag",
    "piqa",
    "mathqa",
]


def main():
    """Main benchmark function."""
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

    # Create compressed model
    wrapper = create_compressed_model(config)
    model = wrapper.get_model_for_training()

    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Create HFLM wrapper for lm_eval
    batch_size = 128
    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_batch_size=batch_size * 2,
    )

    # Run evaluation
    print(f"\nEvaluating on tasks: {BENCHMARK_TASKS}")
    results = lm_eval.simple_evaluate(
        hflm,
        tasks=BENCHMARK_TASKS,
        num_fewshot=0,
        batch_size=batch_size,
        max_batch_size=batch_size * 2,
        device=device,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    if "results" in results:
        for task, metrics in results["results"].items():
            print(f"\n{task}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    main()
