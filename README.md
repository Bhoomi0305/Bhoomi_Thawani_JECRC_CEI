# Mini GPT-2 From Scratch

A miniature **GPT-2 style decoder-only Transformer** implemented and trained from scratch in **PyTorch**. This project is based on the educational lecture and notebook by **Andrej Karpathy** and was completed as a hands-on implementation to understand the internal workings of modern Large Language Models (LLMs).

> **Note:** This project is inspired by Andrej Karpathy's "Let's build GPT" lecture. The implementation follows the educational material while serving as a learning exercise in transformer architectures.

---

## Project Objectives

- Understand decoder-only Transformer architecture
- Implement self-attention from scratch
- Train a mini GPT model on a text corpus
- Learn tokenization, embeddings, positional encoding, residual connections, and autoregressive text generation

---

## Features

- Character-level tokenization
- Multi-Head Self Attention
- Feed Forward Network
- Layer Normalization
- Residual Connections
- Positional Embeddings
- Autoregressive text generation
- Model checkpoint saving
- Training and validation loss monitoring

---

## Project Structure

```text
mini-gpt2-from-scratch/
├── notebook.ipynb
├── README.md
├── requirements.txt
├── outputs/
│   ├── checkpoint.pt
│   ├── generated_samples.txt
│   └── loss_curve.png
└── data/
    └── input.txt
```

---

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- Google Colab

---

## Model Overview

Input Text
→ Token Embeddings
→ Positional Embeddings
→ Transformer Blocks
→ LayerNorm
→ Linear Projection
→ Softmax
→ Next Token Prediction

---

## Dataset

The model is trained on a text dataset (`input.txt`). The default notebook uses Tiny Shakespeare, but any plain-text corpus can be substituted.

---

## Results

The trained model learns language patterns from the dataset and generates coherent text using autoregressive decoding.

Typical outputs improve as training progresses and loss decreases.

---

## Learning Outcomes

- Transformer architecture
- Self-attention mechanism
- Multi-head attention
- Language modeling
- Backpropagation in sequence models
- PyTorch model implementation

---

## Feature

✅ Temperature-based Text Generation

✅ Top-k Sampling

Allows the model to generate text with different creativity levels by adjusting the sampling temperature and restricting predictions to the most probable tokens.

---

## Future Improvements

- Train on larger datasets
- Byte Pair Encoding (BPE) tokenizer
- Multilingual datasets
- Larger GPT configurations
- Alternative attention mechanisms

---

## References

- Andrej Karpathy – _Let's build GPT_
- https://www.youtube.com/watch?v=kCc8FmEb1nY
- https://github.com/karpathy/ng-video-lecture

---

## License

This repository is intended for educational purposes.
