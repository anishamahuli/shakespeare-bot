from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

model = GPT2LMHeadModel.from_pretrained('./shakespeare_gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('./shakespeare_gpt2')

import torch
torch.save(model.state_dict(), "./shakespeare_gpt2/pytorch_model.bin")
model.save_pretrained('./shakespeare_gpt2')
tokenizer.save_pretrained('./shakespeare_gpt2')
print("✅ Model and tokenizer saved to './shakespeare_gpt2'.")