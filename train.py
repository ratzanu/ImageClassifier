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

parser = argparse.ArgumentParser(
    description='Taking user inputs',)

parser.add_argument('data_dir', action ="store", type = str, help = "Input the directory path for data")
parser.add_argument('--save_dir', action ="store", dest = "save_dir" , default = '/home/workspace/ImageClassifier/checkpoint.pth', type = str, help = "Provide path to save the checkpoint")
parser.add_argument('--arch', action ="store",dest = "arch" , default = "vgg16", type = str, help = "Input pretrained model to use")
parser.add_argument('--learning_rate', action ="store",dest = "learning_rate" , default = 0.001, type = int, help = " Define Learning rate")
parser.add_argument('--hidden_units', action ="append",dest = "hidden_layers" , default = [4096,1000] , type = int, help = "Provide hidden layers")
parser.add_argument('--gpu', action ="store",dest = "device" , default = "cuda", type = str, help = "CPU or GPU for learning?")
parser.add_argument('--epochs', action ="store",dest = "epochs" , default = 3, type = int, help = "Number of epochs")


results = parser.parse_args()
device = results.device
epochs = results.epochs


#Load the data from flowers directory

train_dir = results.data_dir + '/train'
valid_dir = results.data_dir + '/valid'
test_dir = results.data_dir + '/test'

#Import the data into dataloaders

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir,   transform = train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir,  transform = valid_test_transforms)
test_datasets = datasets.ImageFolder(test_dir,  transform = valid_test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
test_dataloaders = torch.utils.data.DataLoader(test_datasets)

#Mapping names to class numbers
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Loading Pre-trained model
if results.arch == 'vgg16':
    model = models.vgg16(pretrained = True)
elif results.arch == 'vgg13':
    model = models.vgg13(pretrained = True)
 
# Define Network
hidden_layers = results.hidden_layers

Network(25088 ,102, hidden_layers, model, 0.2)

#Train the model
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = results.learning_rate)

def train_model(traindata, testdata, pretrained_model, device_gpu_cpu, optimizer, epochs, print_every):
    model.to(device_gpu_cpu)
    steps = 0
    
    for e in range(epochs):
        running_loss = 0
        for m,(inputs, labels) in enumerate(traindata):
            steps += 1
        
            inputs, labels = inputs.to(device_gpu_cpu), labels.to(device_gpu_cpu)
        
            optimizer.zero_grad()
        
        # Forward and backward passes
            outputs = pretrained_model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                pretrained_model.eval()
                vloss = 0
                accuracy = 0
                
                for m , (inputv, labelsv) in enumerate(testdata):
                    optimizer.zero_grad()
                    inputv , labelsv = inputv.to(device_gpu_cpu), labelsv.to(device_gpu_cpu)
                    pretrained_model.to(device_gpu_cpu)
                    with torch.no_grad():
                        outputs = pretrained_model.forward(inputv)
                        vloss = criterion(outputs, labelsv)
                        ps = torch.exp(outputs).data
                        comparitor = (labelsv.data == ps.max(1)[1])
                        accuracy += comparitor.type_as(torch.FloatTensor()).mean()
                        
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Current Validation Loss: {:.4f}".format(vloss/len(testdata)),
                      "Accuracy of model: {:.2f}".format(accuracy/len(testdata)))
            
                running_loss = 0
                pretrained_model.train()
                
                
#Train the model

train_model(train_dataloaders, valid_dataloaders, model, device,  optimizer, epochs, print_every = 10)

#Validating the trained model

def validation_function(testdata, pretrained_model, device_gpu_cpu, correct, total):
    model.to(device_gpu_cpu)
    with torch.no_grad():
        for data in testdata:
            images, labels = data
            outputs = pretrained_model(images.to(device_gpu_cpu))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device_gpu_cpu)).sum().item()
            
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

validation_function(test_dataloaders, model, device,  0, 0)
#Save to checkpoint

model.class_to_idx = train_datasets.class_to_idx

checkpoint = {'inputlayer': 25088,
              'outputlayer': 102,
              'hidden_layers': hidden_layers,
              'state_dict':model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'arch' : results.arch,
              'drop_out' : 0.2}


torch.save(checkpoint, results.save_dir)