
import argparse
import numpy as np

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from collections import OrderedDict

import time
import json
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
from Network import Network

# User inputs
parser = argparse.ArgumentParser('User inputs for predict.py')

parser.add_argument('--image_path', default = '/home/workspace/ImageClassifier/flowers/test/1/image_06743.jpg' ,type = str)
parser.add_argument('--checkpoint_path', default = '/home/workspace/ImageClassifier/checkpoint.pth' ,type = str)
parser.add_argument('--topk', default = 5 ,type = int)
parser.add_argument('--gpu', default = 'cuda' ,type = str)
parser.add_argument('--category_names', default = 'cat_to_name.json' ,type = str)

results = parser.parse_args()
device = results.gpu

with open(results.category_names, 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    Network(checkpoint['inputlayer'],checkpoint['outputlayer'],checkpoint['hidden_layers'],model, checkpoint['drop_out'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
  
model = load_checkpoint(results.checkpoint_path)

def process_image(im):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model    
    # Shortest side using thumbnail
    size = (256,256)
    width, height = im.size
    if width > height:
        ratio = float(width) / float(height)
        newheight = ratio * size[0]
        im = im.resize((size[0], int(float(newheight))), Image.ANTIALIAS)
    elif width < height:
        ratio = float(height)/float(width)
        newwidth = ratio * size[1]
        im = im.resize((int(float(newwidth)), size[1]), Image.ANTIALIAS)
    else:
        im = im.resize((size[0], size[1]))
    im = im.crop(coordinates_center_crop(im))
    np_array = np.array(im)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_array = (np_array - mean)/std
    np_array1 = np_array.transpose((2,0,1))
    
    transform  = transforms.ToTensor()
    tensor_array = transform(np_array)

    return tensor_array

def coordinates_center_crop(image):
    xwidth, xheight = image.size
    ywidth, yheight = 224 ,224
    a = (xwidth - ywidth)/2
    b = (xheight - yheight)/2
    c = (xwidth + ywidth)/2
    d = (xheight + yheight)/2
    return (a,b,c,d)

cti = model.class_to_idx
idx_to_class = {val:key for key,val in cti.items()} 

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    im = Image.open(image_path)

    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    img = process_image(im)
    img = img.unsqueeze_(0)
    img = img.float()
    
    with torch.no_grad():
        output = model.forward(img.to(device))
        
    probability = F.softmax(output.data,dim=1)
    probability , indexes = probability.topk(topk)
    
    probability, indexes = probability.to('cpu'), indexes.to('cpu')
    
    probability = probability.detach().numpy()[0]
    indexes = indexes.detach().numpy()[0]
    
    class_label = [idx_to_class[key] for key in indexes]
    class_name = [cat_to_name[key] for key in class_label]
    
    return probability, class_name


def check_sanity(image_path):

    probabilities, classes = predict(image_path, model, results.topk, device)

    print(probabilities)
    print(classes)

check_sanity(results.image_path)
    