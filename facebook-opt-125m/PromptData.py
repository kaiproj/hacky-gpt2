from torch.utils.data import Dataset
import json

# PromptData imports jsonl data with the following format:
#
# {"prompt": "prompt text", "completion": "completion text"}
#
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
            # TODO: Should this trainer be responsible for templating the full
            # string from the prompt and the completion? If so, the trainer might need to be configured
            # with some tokens. See: https://github.com/Pawandeep-prog/finetuned-gpt2-convai
            #
            #s = "<startofstring> "+obj['prompt']+" <completion>: "+obj['completion']+" <endofstring>"
            #
            s = obj['prompt'] + obj['completion']
            #print(s)
            self.X.append(s)

        self.X_encoded = tokenizer(self.X, max_length=10, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])

#mydata = PromptData('./data/sample/favorite-color-blue.jsonl')
