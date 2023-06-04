# Import huggingface transformers and load and save the gpt2 model
# to a file.

import sys
import torch
from transformers import GPT2LMHeadModel

# Load the GPT2 model
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

# Save the model to a file
torch.save(model, 'model.pt')

