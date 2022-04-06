
import pickle
import torch
import torch.nn as nn
import numpy as np

from .dataset import valTransform
from .config import Config
from .model.model import Model

# label number to label name dictionary
with open('/home/ubuntu/Beta_test/nnc/models/model1/data/dictionary.pickle', 'rb') as fr:
    class_dict = pickle.load(fr)

# pre-load model weight for faster inference

path = '/home/ubuntu/Beta_test/nnc/models/model1/weight/model1_epoch18_0.9326717879599149.pth'

args = Config()
args.backbone_pretrained = False
args.device = 'cuda'

model = Model(args)
model.load_state_dict(torch.load(path, map_location=args.device))
model.to(args.device)
model.eval()

def Inference(image, device='cpu'):
    '''

    inference function

    input : PIL.Image
    output : String

    '''
    args = Config()
    args.backbone_pretrained = False
    args.device = device
    
    image = np.array(image.convert('RGB'))
    image = image.copy()
    image = valTransform()(image=image)['image'].unsqueeze(0)
    
    
    #model = Model(args)
    #if path!=None:
    #    model.load_state_dict(torch.load(path, map_location='cpu'))

    #model.to(args.device)
    #model.eval()
    

    with torch.no_grad():
        output = model(image.to(args.device))
        
    clss = output.argmax(1).detach().tolist()[0] 
    confidence = nn.Softmax(dim=-1)(output)[:, clss].detach().tolist()[0]

    print(' ')
    print('confidence: ', confidence)
    print(' ')       
    return class_dict[f'{clss}']