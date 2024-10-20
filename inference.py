import torch
import numpy as np
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel
from load_model import load_model
import os
import argparse

from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stops: List[torch.Tensor], encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.encounters = encounters
        self.counter = {tuple(stop.tolist()): 0 for stop in stops}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        for stop in self.stops:
            if torch.equal(input_ids[0, -len(stop):], stop):
                self.counter[tuple(stop.tolist())] += 1
                if self.counter[tuple(stop.tolist())] >= self.encounters:
                    return True
        return False


class ChatbotEngine:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.max_new_tokens: int = 1024
        self.temperature: float = 0.8
        self.top_p: float = 0.95
        self.top_k: int = 50
        self.repetition_penalty: float = 1.2
        self.model = model
        self.tokenizer = tokenizer

    def _get_prompt(self, messages: List[Dict[str, str]]) -> str:
        system_prompt = "남자와 여자의 대화에서 여자의 말에 공감하면서 대화하세요."
        instruction = system_prompt + '\n\n'
        contexts = "여자: " + messages[0]["content"] + "\n"

        for message in messages[1:]:
            role = message["role"]
            content = message["content"]

            if role == "user":
                contexts += "여자: " + content + "\n"
            elif role == "assistant":
                contexts += "남자: " + content + self.tokenizer.eos_token + "\n"

        prompt = instruction + contexts + "남자: "
        return prompt

    def _get_stopwords_ids(self, stop_words: List[str]) -> List[torch.Tensor]:
        stop_words_ids = [self.tokenizer.encode(stop_word, add_special_tokens=False) for stop_word in stop_words]
        return [torch.tensor(ids) for ids in stop_words_ids]

    def run(self, messages: list[dict[str, str]]) -> str:
        prompt = self._get_prompt(messages)

        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            add_special_tokens=False,
            return_token_type_ids=False,
        )

        stop_words_ids = self._get_stopwords_ids(self.tokenizer.eos_token)
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stops=stop_words_ids)])

        generate_kwargs = dict(
            input_ids=inputs["input_ids"],
            max_length=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            num_beams=1,
            top_p=self.top_p,
            top_k=self.top_k,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=self.repetition_penalty,
        )

        outputs = self.model.generate(**generate_kwargs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response[response.rfind("남자:") + 3:].strip()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--model_cache", type=str, default=None)
    parser.add_argument("--lora_config", type=str, default=None)

    return parser.parse_args()


def set_cuda_device(device: str):
    device_ids = device.split(',')
    if all(dev.isdigit() for dev in device_ids):
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        print(f"Using CUDA device:{device}")
    else:
        raise ValueError("Invalid device index. Must be a numeric value.")


def main():
    args = parse_arguments()
    set_cuda_device(args.device)

    base_model, tokenizer = load_model(cache_dir=args.model_cache)
    model = PeftModel.from_pretrained(base_model, args.lora_config)
    chatbot = ChatbotEngine(model, tokenizer)

    messages = []

    while True:
        user_input = input("여친: ")
        if(user_input == "그만해"):
            break

        messages.append({"role": "user", "content": user_input})
        response = chatbot.run(messages)
        print("남친:", response)

        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()