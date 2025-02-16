<h1 align="center">LLM DIY KIT</h1>
<p align="center">
  <img width="150" src="./assets/diy.png" />
</p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QetATApxZnSxEDMMJkvOhVdFX6tYMl_h?usp=sharing)

This project has been created to help you understand how large language models (LLMs) work. It provides a fundamental language model and tokenizer along with scripts for pretraining, instruction fine-tuning (SFT) using LoRA, and more.

---

## Project Structure

```
.
├── README.md
├── assets
│   └── diy.png
├── model
│   ├── __init__.py
│   ├── decoder.py
│   ├── linear.py
│   ├── mlm_head.py
│   ├── transformer.py
│   └── transformer_block.py
├── requirements.txt
├── settings.py
├── sft
│   ├── __init__.py
│   ├── dataset.py
│   └── train.py
├── tokenizer
│   ├── __init__.py
│   └── tokenizer.py
└── train
    ├── __init__.py
    ├── dataset.py
    └── trainer.py
```

---

## Model Architecture

```mermaid
flowchart TD
    subgraph TransformerModel
      A[Input Tokens] --> B[Decoder]
      B --> C[LM Head (Linear)]
      C --> D[Output Logits]
    end

    subgraph Decoder [Decoder]
      direction TB
      E[Token Embedding] --> F[Position Embedding]
      G[Sum Embeddings] --> H[Stack of Transformer Blocks]
      E --> G
      F --> G
      G --> H
    end

    subgraph TransformerBlock [Transformer Block]
      direction LR
      I[Multi-Head Self-Attention]
      J[Add & Norm]
      K[Feed-Forward Network<br/>(with LoRALinear modules)]
      L[Add & Norm]
      I --> J
      J --> K
      K --> L
    end

    %% Connect Decoder to multiple TransformerBlocks
    H --> TransformerBlock
```

---

## Introduction

LLM DIY KIT offers a minimalist implementation of a language model built from scratch. This repository includes:

- A decoder-only Transformer model.
- A custom GPT-2 based tokenizer.
- Pretraining on a large textual corpus (e.g., Simple Wikipedia).
- Instruction fine-tuning using LoRA for efficient adaptation.
- Example training scripts and datasets for both pretraining and fine-tuning.

---

## Introduction

LLM DIY KIT offers a minimalist implementation of a language model built from scratch. This repository includes:

- A decoder-only Transformer model.
- A custom GPT-2 based tokenizer.
- Pretraining on a large textual corpus (e.g., Simple Wikipedia).
- Instruction fine-tuning using LoRA for efficient adaptation.
- Example training scripts and datasets for both pretraining and fine-tuning.

---

## Getting Started

### Step 1: Install Dependencies

Install the required packages:

```bash
pip3 install -r requirements.txt
```

### Step 2: Pretrain the Model

Pretrain the baseline Transformer model by running:

```bash
PYTHONPATH=$(pwd) python3 train/trainer.py
```

This script trains the model on the Simple Wikipedia dataset (loaded automatically via HuggingFace Datasets) and saves the pretrained weights to `baseline_transformer.pth`.

### Step 3: Instruction Fine-Tuning (SFT) with LoRA

Fine-tune the pretrained model using LoRA on an instruction dataset by running:

```bash
PYTHONPATH=$(pwd) python3 sft/train.py
```

This script loads the pretrained weights, applies LoRA (with only the additional low-rank parameters being trainable), and saves the fine-tuned model to `lora_sft_transformer.pth`.

---

## Tokenizer

The project utilizes a GPT-2 based tokenizer from the [Transformers library](https://huggingface.co/docs/transformers/en/index). The tokenizer is configured to use the `eos_token` as the pad token to properly handle padding during training.

---

## Future Work

- Extend fine-tuning guidelines with further architecture updates.
- Increase model parameters and benchmark performance.
- Provide prediction examples.
- Include detailed model architecture descriptions along with video tutorials.

---

## License

This project is licensed under the MIT License.
