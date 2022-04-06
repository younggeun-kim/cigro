
import pickle
import torch
import torch.nn as nn
import numpy as np
from transformers import ElectraModel, ElectraTokenizer

from config import Config
from model.model import Model

# label number to label name dictionary
#with open('/home/ubuntu/Beta_test/nnc/models/model1/data/dictionary.pickle', 'rb') as fr:
#    class_dict = pickle.load(fr)

# pre-load model weight for faster inference

path = '/root/yg/cigro/BETA/models/model1/output/train/20220402-130031-resnext50_32x4d/checkpoint-6.pth'

args = Config()
args.backbone_pretrained = False
args.device = 'cuda'

model = Model(args)
model.load_state_dict(torch.load(path, map_location=args.device)['state_dict'])
model.to(args.device)
model.eval()

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

def tokenize(sequences):
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

class_dict = {
    '0': '굿티',
    '1': '보고쿡',
    '2': '피피빅',
    '3': '바디럽',
    '4': '모도리',
    '5': '공백',
    
}

def Inference(text, device='cpu'):
    '''

    inference function

    input : PIL.Image
    output : String

    '''
    args = Config()
    args.backbone_pretrained = False
    args.device = device

    input_ids, attention_masks = tokenize([text])
    inputs = [torch.as_tensor(input_ids, dtype=torch.long).unsqueeze(0).to(args.device),\
         torch.as_tensor(attention_masks, dtype=torch.long).unsqueeze(0).to(args.device)]
    #input = [x.to(args.device) for x in inputs]
    #model = Model(args)
    #if path!=None:
    #    model.load_state_dict(torch.load(path, map_location='cpu'))

    #model.to(args.device)
    #model.eval()
    

    with torch.no_grad():
        output = model(inputs)
        
    clss = output.argmax(1).detach().tolist()[0] 
    confidence = nn.Softmax(dim=-1)(output)[:, clss].detach().tolist()[0]

    print(' ')
    print('confidence: ', round((confidence * 100),2), "%")
    print(' ')       
    return class_dict[f'{clss}']