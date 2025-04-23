import types
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from contextlib import contextmanager
from abc import abstractmethod
import torch.nn as nn
import torch
import vigilator

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
    shape = logits.shape
    assert len(shape) == 3
    reshaped_logits = logits.reshape(1, shape[1] * shape[2])
    return reshaped_logits.tolist()[0]

@contextmanager
def add_vigilator(model: Model, tokenizer_vocab: dict):
    server_is_launched: bool = False
    def call_wrapper(func):
        def wrapper(self, *args, **kwargs):
            input_tokens = kwargs.get("input_ids")
            output = func(*args, **kwargs)
            assert type(input_tokens) == torch.Tensor
            vigilator.send_output(input_ids_to_list(input_tokens), logits_to_list(output.logits))
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


    with add_vigilator(model, tokenizer.vocab):

        inputs = tokenizer("Hello, I am", return_tensors="pt")
        tokens = model.__call__(**inputs)

    print(vigilator.retrieve_output(1))

if __name__ == "__main__":
    main()