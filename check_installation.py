import transformers
import datasets
import torch

print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
print(f"Torch version: {torch.__version__}")
print(f"MPS Available: {torch.backends.mps.is_available()}")