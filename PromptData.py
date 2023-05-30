# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---



# +
from torch.utils.data import Dataset
import json

class PromptData(Dataset):
    def __init__(self, path:str, tokenizer):
        with open(path, 'r') as json_file:
            json_list = list(json_file)

        self.data = []
        for json_str in json_list:
            obj = json.loads(json_str)
            self.data.append(obj)

        self.X = []
        for obj in self.data:
            s = "<startofstring> "+obj['prompt']+" <completion>: "+obj['completion']+" <endofstring>"
            print(s)
            self.X.append(s)

        #self.X = self.X[:5000]

        self.X_encoded = tokenizer(self.X, max_length=10, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])

# +
#mydata = PromptData('./data/sample/favorite-color-blue.jsonl')
# -


