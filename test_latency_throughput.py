"""
Latency and throughput benchmarking for Basis Sharing compressed models.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizer

from config import ShareConfig, add_args
from model_factory import create_compressed_model
from prepare_data import prepare_data


def main():
    """Main benchmarking function."""
    cmd_args = add_args()
    config = ShareConfig(cmd_args)

    print(f"Model: {config.model_name}")
    print(f"Compression ratio: {config.compression_ratio}%")

    # Load tokenizer
    if config.model_type == "llama2":
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.pad_token or "[PAD]"

    # Load dataset
    seq_length = 32
    _, val_dataset, _, data_collator = prepare_data(
        config.dataset_name,
        tokenizer,
        seq_length,
        getattr(config, "dataset_cache_dir", None),
    )

    # Create compressed model
    wrapper = create_compressed_model(config)
    model = wrapper.get_model_for_training()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.config.use_cache = False
    model.eval()

    print("Compiling model with torch.compile...")
    model = torch.compile(model)

    # Create dataloader
    batch_size = 512
    indices = list(range(len(val_dataset) // 4))
    subset = Subset(val_dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4,
    )

    # Benchmark
    total_times = []
    warmup_done = False

    for batch in tqdm(dataloader, desc="Benchmarking"):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}

            # Warmup
            if not warmup_done:
                print("Warming up...")
                for _ in range(10):
                    _ = model(**batch)
                warmup_done = True

            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(**batch)

            if device == "cuda":
                torch.cuda.synchronize()

            total_times.append(time.perf_counter() - start)

    # Results
    tokens_per_batch = batch_size * seq_length
    median_time = np.median(total_times)
    throughput = tokens_per_batch / median_time

    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Median time per batch: {median_time * 1000:.2f} ms")
    print(f"Throughput: {throughput:,.0f} tokens/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
