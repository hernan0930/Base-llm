
# Base_LLM: GPT-2 from Scratch

Base_LLM is a minimal, educational implementation of a GPT-2 language model from scratch in PyTorch. This project demonstrates the core components of transformer-based language models, including data loading, tokenization, model architecture, and training loop, using a small dataset for clarity and learning purposes.

## Features
- **Custom GPT-2 Model**: Implements a simplified GPT-2 architecture in `models/gpt2_model.py`.
- **Flexible Data Pipeline**: Loads and tokenizes text data using OpenAI's `tiktoken` GPT-2 tokenizer.
- **Training & Validation**: Includes a training loop with periodic evaluation and loss tracking.
- **Educational Structure**: Modular code for easy understanding and experimentation.

## Project Structure

```
Base_LLM/
  main.py                # Main training script
  utils.py               # Utility functions (e.g., dataloader)
  data/
    dataloader_gpt.py    # Dataloader implementation
    load_data.py         # Data loading utilities
  models/
    gpt2_model.py        # GPT-2 model definition
  training/
    training.py          # Training loop and helpers
  README.md
```

## Quick Start
1. **Install requirements** (PyTorch, tiktoken):
   ```bash
   pip install torch tiktoken
   ```
2. **Run the main script:**
   ```bash
   python main.py
   ```

## How It Works
- Downloads a sample text dataset.
- Splits data into training and validation sets.
- Tokenizes text using GPT-2 tokenizer.
- Trains a GPT-2 model from scratch and evaluates periodically.

## Credits
Inspired by [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).

---
For learning and research purposes only.
