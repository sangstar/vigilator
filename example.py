import time

import requests
import types
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from contextlib import contextmanager
from abc import abstractmethod
import torch.nn as nn
import torch
import vigilator
from dataclasses import dataclass

class ModelOutput:
    uuid: str
    timestamp: str
    text: str
    token_ids: list[int]
    logits: torch.Tensor

    def __init__(self,
                 uuid,
                 timestamp,
                 text,
                 token_ids,
                 logits):
        self.uuid = uuid
        self.timestamp = timestamp
        self.text = text
        self.token_ids = token_ids
        self.logits = torch.Tensor(logits)


class Model:
    model: nn.Module

    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def __call__(self, input_tokens: torch.Tensor) -> torch.Tensor:
        pass

class TransformersModel(Model):

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def __call__(self, input_tokens: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_tokens)

def input_ids_to_list(tens: torch.Tensor) -> list:
    return tens.tolist()[0]

def logits_to_list(logits: torch.Tensor) -> list:
    # Only concern self with last forward pass for logits for next token
    return logits[:, -1, :].tolist()[0]

@contextmanager
def add_vigilator(model: Model, tokenizer_vocab: dict):
    server_is_launched: bool = False
    def call_wrapper(func):
        def wrapper(self, *args, **kwargs):
            text = kwargs.get("text")
            input_tokens = kwargs.get("input_ids")
            output = func(*args, **kwargs)
            assert type(input_tokens) == torch.Tensor
            vigilator.send_output(text, input_ids_to_list(input_tokens), logits_to_list(output.logits))
            return output
        return wrapper

    original_generate_method = model.__call__
    new_generate_method = call_wrapper(original_generate_method)
    try:
        model.__call__ = types.MethodType(new_generate_method, model)
        yield
    finally:
        model.__call__ = original_generate_method





def main():

    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m",
    )

    prompt = "Hello, I am"

    vigilator.start_server()

    with add_vigilator(model, tokenizer.vocab):

        inputs = tokenizer(prompt, return_tensors="pt")
        req_dict = {**inputs}
        req_dict["text"] = prompt
        tokens = model.__call__(**req_dict)

    time.sleep(5)
    response = requests.get("http://localhost:3000/", params={"text": prompt, "top_k": 5})
    if response.status_code == 200:
        data = response.json()
        print(
            ModelOutput(
                uuid="",
                timestamp="",
                text=data["text"],
                token_ids=data["token_ids"],
                logits=data["logits"],
            )
        )
    else:
        raise Exception(f"Failed to retrieve output: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()