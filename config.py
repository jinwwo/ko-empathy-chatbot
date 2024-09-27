from transformers import TrainingArguments, DataCollatorForLanguageModeling
from typing import Optional
from peft import LoraConfig


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    bias: str = "none",
    use_rslora: bool = False
):
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias=bias,
        use_rslora=use_rslora,
    )


def get_training_arguments(
        output_dir: str = "outputs",
        num_epochs: int = 1,
        learning_rate: float = 2e-4,
        lr_scheduler_type: str = "cosine",
        logging_steps: int = 500,
        # eval_steps: int = 500,
        batch_size: int = 1
):
    return TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        fp16=True,
        output_dir=output_dir,
        save_total_limit=2,
        logging_steps=logging_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        # evaluation_strategy="steps",
        # eval_steps=eval_steps,
        report_to=["tensorboard"],
    )