import json
import os

import argparse
from datasets import Dataset, DatasetDict
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling, Trainer

from config import get_lora_config, get_training_arguments
from load_model import load_model
from utils import process_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_model_for_training(model, lora_config):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False)

    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    model, tokenizer = load_model(cache_dir='/home/jinuman/chat/models')
    model = prepare_model_for_training(model, lora_config=get_lora_config())
    print_trainable_parameters(model)

    with open("/home/jinuman/chat/ko-couple-chat.json", 'r') as file:
        data = json.load(file)
    train_dataset = Dataset.from_dict(data)
    dataset = DatasetDict({
        'train': train_dataset
    })
    dataset = process_data(dataset, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    training_args = get_training_arguments(
        num_epochs=2, batch_size=1
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        # eval_dataset=dataset['validation']
        data_collator=data_collator
    )

    trainer.train(resume_from_checkpoint="/home/jinuman/chat/outputs/checkpoint-6000")
    model.save_pretrained(args.model_path)


if __name__ == "__main__":
    train()