from typing import Dict, Optional, Union
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

_DEFAULT_MODEL_ID = "yanolja/EEVE-Korean-10.8B-v1.0"

def load_model(
    model_id: Optional[str] = None,
    quantization: bool = True,
    device_map: Union[int, str] = "auto",
    model_kwargs: Optional[Dict] = None,
    cache_dir: Optional[str] = None,
):
    model_id = model_id or _DEFAULT_MODEL_ID
    model_kwargs = model_kwargs or {}
    bnb_config = None

    kwargs = {
        "pretrained_model_name_or_path": model_id,
        "device_map": device_map,
        "trust_remote_code": True,
        "cache_dir": cache_dir
    }
    kwargs.update(model_kwargs)

    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs.update({"quantization_config": bnb_config})
        
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(**kwargs)

    return model, tokenizer