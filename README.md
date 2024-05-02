<h1 align="center">LLM DIY KIT</h1>

This project has been made to help to understand how LLM works.

## Introduction

This project includes fundamental language model and tokenizer for pre-training from the zero.

### Getting started

Download dataset first.

```bash
sh download_dataset.sh
```

Install the packages.

```bash
pip3 install -r requirements.txt
```

You can start the train now on.

```bash
PYTHONPATH=$(pwd) python3 train/trainer.py
```

### Tokenizer

This model uses [XLMRobertaTokenizer](https://huggingface.co/docs/transformers/en/model_doc/xlm-roberta) of [Transformers](https://huggingface.co/docs/transformers/en/index).
