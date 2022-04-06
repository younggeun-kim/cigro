import timm

import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraTokenizer



class Model(nn.Module):


    def __init__(self, config):
        super(Model, self).__init__()
        self.model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator") 
        self.dropout = nn.Dropout(p=config.drop_rate)
        self.fc = nn.Linear(config.feature_dim, config.num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight) 
        
    def forward(self, x):
        x = self.model(x[0], attention_mask=x[1],)[0][:,0,:]
        x = self.dropout(x)
        return self.fc(x)
