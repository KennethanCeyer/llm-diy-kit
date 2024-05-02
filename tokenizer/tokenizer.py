from transformers import XLMRobertaTokenizer


class Tokenizer:
    def __init__(self):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    def tokenize(self, text, return_tensors="pt", **kwargs):
        return self.tokenizer(text, return_tensors=return_tensors, **kwargs)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
