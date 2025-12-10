"""
LoRA fine-tuning for Basis Sharing compressed models.
"""

from __future__ import annotations

import os

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, LlamaTokenizer, Trainer, TrainingArguments

from basis_sharing import BasisSharingWrapper
from config import ShareConfig, add_args
from model_factory import create_compressed_model
from prepare_data import prepare_data
from test import compute_ppl


def main():
    """Main LoRA fine-tuning function."""
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

    # Load datasets
    train_dataset, val_dataset, test_dataset, data_collator = prepare_data(
        config.dataset_name,
        tokenizer,
        config.context_length,
        getattr(config, "dataset_cache_dir", None),
    )

    # Create compressed model
    wrapper = create_compressed_model(config)
    model = wrapper.get_model_for_training()

    # LoRA output directory
    lora_output_dir = getattr(config, "lora_output_dir", "./lora_output")

    if os.path.exists(lora_output_dir):
        print(f"Loading existing LoRA weights from {lora_output_dir}")
        model = PeftModel.from_pretrained(model, lora_output_dir)
    else:
        print("Starting LoRA fine-tuning...")

        # Set wandb project name
        os.environ["WANDB_PROJECT"] = ShareConfig.name_map.get(
            config.model_name, config.model_name
        )

        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=getattr(config, "lora_r", 8),
            lora_alpha=getattr(config, "lora_alpha", 32),
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Training arguments
        trainer_config = TrainingArguments(
            output_dir=lora_output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=getattr(config, "lora_train_batch_size", 1),
            per_device_eval_batch_size=getattr(config, "lora_train_batch_size", 1),
            gradient_accumulation_steps=1,
            lr_scheduler_type="constant",
            logging_steps=1,
            learning_rate=getattr(config, "lora_learning_rate", 1e-4),
            save_total_limit=1,
            seed=42,
            data_seed=0,
            save_safetensors=False,
            bf16=True,
            num_train_epochs=getattr(config, "lora_train_epoch", 2),
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
        )

        # Train
        trainer = Trainer(
            model=model,
            args=trainer_config,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
        )

        trainer.train()

        if getattr(config, "save_lora", True):
            trainer.save_model()
            print(f"LoRA weights saved to {lora_output_dir}")

    # Evaluate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppl = compute_ppl(config.context_length, config.stride, test_dataset, model, device)
    print(f"\nPerplexity after LoRA: {ppl.item():.2f}")


if __name__ == "__main__":
    main()
