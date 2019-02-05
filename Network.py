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


# User inputs
import argparse



def Network(inputlayer,outputlayer,hidden_layers,pretrained_model,drop_out):
    for param in pretrained_model.parameters():
        param.requires_grad = False
    input_layer = nn.Linear(inputlayer, hidden_layers[0])
    output_layer = nn.Linear(hidden_layers[-1], outputlayer)
    

    classifier = nn.Sequential(OrderedDict([
                              ('fc_input:', input_layer),
                              ('relu:', nn.ReLU()),
                              ('Dropout:', nn.Dropout(drop_out)),
                              ('fc2:', nn.Linear(hidden_layers[0], hidden_layers[-1])),
                              ('relu:', nn.ReLU()),
                              ('Dropout:', nn.Dropout(drop_out)),
                              ('fc_output:', output_layer),
                              ('output:', nn.LogSoftmax(dim=1))
                              ]))
    
    pretrained_model.classifier = classifier
    return pretrained_model