import torch
from transformers import pipeline, GPT2Tokenizer

from fastapi import FastAPI, Response
from pydantic import BaseModel

# load model
model = torch.load("model.pt")
# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

# create pipeline
generate = pipeline("text-generation",model=model,tokenizer=tokenizer)#,device=0)

# run prediction
print(generate("Testing generation works... ", max_length=30))

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 30

@app.get("/")
def root():
  return Response("Hello there!")

@app.post("/generate")
def generate_text(req: GenerationRequest):
  return generate(req.prompt, max_length=req.max_length)[0]['generated_text']
