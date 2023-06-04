# Download huggingface transformers and load and save the model and tokenizer to disk for later inference and finetuning.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

torch.save(model, '/built/model.pt')
torch.save(tokenizer, '/built/tokenizer.pt')

