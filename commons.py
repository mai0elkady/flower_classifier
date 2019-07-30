import io

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from collections import OrderedDict
import torchvision.transforms as transforms


def get_model():
    checkpoint_path = 'flower_classifier_2.pth'
    checkpoint = torch.load(checkpoint_path,map_location='cpu')
    if(checkpoint['arch'] == 'densenet161'):
        model = models.densenet161(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  
    model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2208, 1024)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1024, 512)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'],strict = False)
    model.eval()
    return model

def get_tensor(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(256),
        				    transforms.CenterCrop(224),
        				    transforms.ToTensor(),
        				    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             					  std=[0.229, 0.224, 0.225])])
                                                  
 
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

