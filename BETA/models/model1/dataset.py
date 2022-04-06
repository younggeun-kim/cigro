import torch.utils.data as data
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import torch
from transformers import ElectraModel, ElectraTokenizer
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

class Dataset(data.Dataset):
    def __init__(self, args='', ann_files=''):
        super(Dataset, self).__init__()
        self.args = args
        self.ann_files = ann_files
        
    def __getitem__(self, index):
        ann_file = self.ann_files.loc[index]
        text = ann_file['text']
        input_ids, attention_masks = self.tokenize([text])
        inputs = (torch.as_tensor(input_ids, dtype=torch.long), torch.as_tensor(attention_masks, dtype=torch.long))

        target = torch.tensor([int(ann_file['label'])], dtype=torch.long)
                
        return inputs, target
    
    def __len__(self):
        return len(self.ann_files)


    def tokenize(self, sequences):
        input_ids = []
        attention_masks = []

        for seq in sequences:
            encoded_dict = tokenizer.encode_plus(
                seq,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=32,  # Pad & truncate all sentences.
                truncation=True,
                padding='max_length',
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids.squeeze(), attention_masks.squeeze()