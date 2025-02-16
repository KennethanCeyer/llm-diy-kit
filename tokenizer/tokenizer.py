from transformers import GPT2TokenizerFast


class Tokenizer:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(self, text, return_tensors="pt", **kwargs):
        return self.tokenizer(text, return_tensors=return_tensors, **kwargs)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
