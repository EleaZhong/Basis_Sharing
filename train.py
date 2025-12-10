"""
Training script for fine-tuning Basis Sharing compressed models.
"""

from __future__ import annotations

from transformers import AutoTokenizer, LlamaTokenizer, Trainer, TrainingArguments, set_seed

from basis_sharing import BasisSharingWrapper
from config import ShareConfig, add_args
from model_factory import create_compressed_model
from prepare_data import prepare_data


def main():
    """Main training function."""
    cmd_args = add_args()
    config = ShareConfig(cmd_args)
    set_seed(2024)

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
    train_dataset, val_dataset, _, data_collator = prepare_data(
        config.dataset_name,
        tokenizer,
        config.context_length,
        getattr(config, "dataset_cache_dir", None),
    )

    # Create compressed model
    wrapper = create_compressed_model(config)
    model = wrapper.get_model_for_training()

    # Setup training arguments
    output_dir = getattr(config, "compressed_model_path", "./trained_model")
    trainer_config = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        lr_scheduler_type="constant",
        logging_steps=1,
        learning_rate=2e-6,
        save_total_limit=1,
        seed=42,
        data_seed=0,
        weight_decay=0.001,
        max_grad_norm=0.01,
        bf16=True,
        num_train_epochs=3,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        run_name=f"basis_sharing_{config.model_type}",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=trainer_config,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save
    trained_model = trainer.model
    trained_model.save_pretrained(
        output_dir,
        safe_serialization=False,
        is_main_process=trainer.accelerator.is_main_process,
        save_function=trainer.accelerator.save,
        state_dict=trainer.accelerator.get_state_dict(trained_model, unwrap=False),
    )

    wrapper_path = f"{output_dir}/wrapper_state.pt"
    wrapper.save_compressed(wrapper_path)

    print(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
