# Basis Sharing

Model-agnostic cross-layer parameter sharing for Large Language Model compression.

This repository implements basis sharing compression, which reduces model size by sharing basis matrices across grouped transformer layers while maintaining per-layer coefficients.

## Architecture

The **model-agnostic** architecture uses a wrapper-based approach that works with any HuggingFace transformer model:

```python
from transformers import AutoModelForCausalLM
from basis_sharing import BasisSharingWrapper, BasisSharingConfig, compress_model

# Load any transformer model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure compression
config = BasisSharingConfig(
    target_modules=["self_attn.*_proj", "mlp.*_proj"],  # Regex patterns
    group_size=2,           # Layers per group
    compression_ratio=0.2,  # 20% of original size
)

# Wrap and compress
wrapper = BasisSharingWrapper(model, config)
wrapper.calibrate(dataloader)  # Collect input statistics
wrapper.compress()             # Run SVD and setup weight tying

# Use for inference
outputs = wrapper(input_ids=input_ids)

# Or use the one-liner
wrapper = compress_model(model, dataloader, target_modules=["self_attn.*_proj"])
```

### Key Components

- **`BasisSharingWrapper`**: Wraps any model and manages the compression pipeline
- **`ShareLinear`**: Drop-in replacement for `nn.Linear` with basis sharing support
- **`BasisSharingConfig`**: Configuration for target modules, grouping, and compression ratio

### Supported Models

Works automatically with:
- LLaMA / LLaMA-2
- Mistral
- OPT
- GPT-2
- Any HuggingFace transformer with `nn.Linear` attention/MLP layers

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Using YAML Config

```bash
# Compress and evaluate on WikiText
python test.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama2_7b_20.yaml
```

### Using Python API

```python
from basis_sharing import compress_model
from torch.utils.data import DataLoader

wrapper = compress_model(
    model=model,
    dataloader=calibration_dataloader,
    target_modules=["self_attn.*_proj", "mlp.*_proj"],
    group_size=2,
    compression_ratio=0.2,
)

# Save/load compressed model
wrapper.save_compressed("compressed_model.pt")
wrapper = BasisSharingWrapper.load_compressed(model, "compressed_model.pt")
```

## Evaluation

### Perplexity

```bash
python test.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama2_7b_20.yaml
```

Test with different datasets:
```bash
python test.py --cf <config.yaml> --dataset_name <ptb|c4|wikitext>
```

### Reasoning Tasks (lm_eval)

```bash
python test_adapter.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama2_7b_20.yaml
```

### Throughput Benchmarking

```bash
python test_latency_throughput.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama2_7b_20.yaml
```

## Fine-tuning

### LoRA

```bash
python lora.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama2_7b_20.yaml
```

### Full Fine-tuning

```bash
python train.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama2_7b_20.yaml
```

## Configuration

Example YAML:

```yaml
model_args:
  model_type: "llama2"
  model_name: "meta-llama/Llama-2-7b-hf"
  group_size: 2
  compression_ratio: 20
  context_length: 2048
  stride: 2048

calibration_args:
  dataset_name: "wikitext"
  calibration_size: 256
  calib_batch_size: 16
  dataset_cache_dir: null

model_saving:
  save_compressed_model: true
  compressed_model_path: "./compressed_models/llama2_7b_20"
```

## How It Works

1. **Replace Linear Layers**: Target `nn.Linear` modules are replaced with `ShareLinear`
2. **Calibration**: Forward passes collect input statistics (X^T @ X) for whitening
3. **Grouping**: Layers are grouped (e.g., layers 0-1, 2-3, etc.)
4. **SVD Compression**: For each group:
   - Compute whitening matrix from calibration data
   - Concatenate weights from grouped layers
   - Apply whitened SVD: W' = S @ W, then SVD(W') = U Σ V^T
   - Shared basis: B = S^{-1} @ U @ Σ (weight-tied across group)
   - Per-layer coefficients: C_i = V^T split by layer
5. **Inference**: Forward pass uses `coefficient(basis(x))`

## File Structure

```
basis_sharing/
├── basis_sharing.py      # Core: BasisSharingWrapper, ShareLinear, Config
├── model_factory.py      # Model creation and compression utilities
├── config.py             # YAML configuration loading
├── calib.py              # Calibration utilities
├── group.py              # Grouping and SVD utilities
├── utils.py              # General utilities
├── test.py               # Perplexity evaluation
├── train.py              # Fine-tuning script
├── lora.py               # LoRA fine-tuning
├── test_adapter.py       # lm_eval benchmarks
├── test_latency_throughput.py  # Performance benchmarks
└── tasks/configs/        # YAML configurations
```

## Reference

```bibtex
@misc{parametersharing2024,
  title={Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression},
  author={Jingcun Wang and Yu-Guang Chen and Ing-Chao Lin and Bing Li and Grace Li Zhang},
  archivePrefix={arXiv},
  year={2024}
}
```
