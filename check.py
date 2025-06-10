import torch
import torchvision
from torch.distributed.tensor import DTensor
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("DTensor available:", hasattr(torch.distributed.tensor, "DTensor"))

# Just test model loading (no training here)
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print("Transformers model and tokenizer loaded successfully.")
