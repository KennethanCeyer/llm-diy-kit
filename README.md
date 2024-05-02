<h1 align="center">LLM DIY KIT</h1><p align="center"><img width="150" src="./assets/diy.png" /></p>

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

## Future work

- Fine-tuning guideline with the architecture updates.
- Parameter increasement with benchmarks.
- Prediction example.
- Model architecture description with video.
