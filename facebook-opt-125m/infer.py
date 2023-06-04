import sys
import torch

class Inferer():
    def __init__(self, model, tokenizer, device):
      self.model = model
      self.tokenizer = tokenizer
      self.device = device
    def infer(self, inp, max_new_tokens):
        inp = self.tokenizer(inp, return_tensors="pt")
        X = inp["input_ids"].to(self.device)
        a = inp["attention_mask"].to(self.device)
        output = self.model.generate(X, attention_mask=a, max_new_tokens=max_new_tokens)
        output = self.tokenizer.decode(output[0])
        return output

if __name__ == "__main__":
  model = torch.load('/built/model.pt')
  tokenizer = torch.load('/built/tokenizer.pt')
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  print(Inferer(model, tokenizer, device).infer(sys.argv[1], 20))
