import json
import os
from datasets import Dataset, DatasetDict


def process_data(
    dataset,
    tokenizer,
    max_length: int = 2048
):
    EOS_TOKEN = tokenizer.eos_token
    dataset = dataset.map(
        lambda x: {
            "text": "\n".join(
                [
                    f"{'여자' if line['role']=='speaker' else '남자'} : {line['text']}{EOS_TOKEN if line['role']!='speaker' else ''}"
                    for line in x["conversations"]
                ]
            )
        }
    )
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=max_length),
        batched=True
    )
    return dataset