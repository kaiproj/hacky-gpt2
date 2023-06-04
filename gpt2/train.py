from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PromptData import PromptData
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch

import sys

p = sys.argv[1]

# +
def train(dat, model, optim):

    epochs = 10

    for i in tqdm.tqdm(range(epochs)):
        for X, a in dat:
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
        #torch.save(model.state_dict(), "model_state.pt")
        print(infer("What is your favorite color?"))

def infer(inp):
    inp = "<startofstring> "+inp+" <completion>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a)
    output = tokenizer.decode(output[0])
    return output


# -

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# +
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<completion>:"])

#model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model = torch.load('./model.pt')
model.resize_token_embeddings(len(tokenizer))
# -

ds = PromptData(p, tokenizer)

dl = DataLoader(ds, batch_size=1)

# +

model.train()

optim = Adam(model.parameters(), lr=1e-3)

print("training .... ")
train(dl, model, optim)

#torch.save(model, '/models/trained.pt')
torch.save(model, './model.pt')

# -
infer("What is your favorite color?")

