
# MiniGPT–README

This project is a minimal, educational implementation of a small GPT-style language model in PyTorch.
It’s meant for understanding transformers, causal attention, and text generation.

# Features
* Byte-level tokenizer (0–255)
* Full transformer architecture (causal attention, MLP, blocks)
* Text generation with temperature, top-k, and top-p
* Trainable on any UTF-8 text file
* Save/load model weights

# Training

Prepare a text file like `data.txt`, then run:

```bash
python mini_gpt.py --train --data data.txt
```

The trained weights will be saved as `mini_gpt.pth`.

## Sampling

After training, generate text with:

```bash
python mini_gpt.py --sample --prompt " Hello, What's up? "
```

## Configuration
All model parameters (e.g., `n_layers`, `n_heads`, `n_embd`, `block_size`) are defined in the `Config` class and can be modified.

## Notes
This is a small model; better quality requires larger configs, more data, and longer training.
