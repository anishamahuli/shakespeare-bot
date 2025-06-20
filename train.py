from datasets import load_from_disk
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load the dataset
lm_dataset = load_from_disk('lm_dataset')

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS backend")
else:
    device = torch.device("cpu")
    print("Using CPU")

model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy='steps',
    eval_steps=100,
    save_steps=500,
    logging_steps=50,
    learning_rate=5e-4,
    warmup_steps=100,
    save_total_limit=2,
    fp16=False,  # MPS backend currently doesn't support fp16
)

from transformers import DataCollatorForLanguageModeling

# Data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    eval_dataset=lm_dataset,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()
print("✅ Training complete. Saving model...")
trainer.save_model('./shakespeare_gpt2')
tokenizer.save_pretrained('./shakespeare_gpt2')
print("✅ Model saved.")