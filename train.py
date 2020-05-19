import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.utils.data

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from os import listdir
import time
import copy
import argparse

#Default Variables
arch='densenet121'
hidden_units=320
learning_rate=0.003
epochs = 3
device = 'cpu'

# Define a parser
parser = argparse.ArgumentParser()

#Chose architecture
parser.add_argument('-a','--arch',
                    action='store',
                    type=str,
                    help='Choose any densenet architecture from torchvision.models')

#Choose Hidden Units
parser.add_argument('-H','--hidden_units',
                    action='store',
                    type=int,
                    help='Choose number of hidden units')

# Add checkpoint directory to parser
parser.add_argument('-s','--save_dir',
                    type=str,
                    help='Choose name of file to save trained model.')

# Add hyperparameter tuning to parser
parser.add_argument('-l','--learning_rate',
                    type=float,
                    help='Choose gradient descent learning rate')

parser.add_argument('-e','--epochs',
                    type=int,
                    help='Choose number of epochs')

parser.add_argument('-g','--gpu',
                    action='store_true',
                    help='Use GPU if available')


# Parse args
args = parser.parse_args()

if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.gpu:
    device = 'cuda'


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
image_datasets = {}
image_datasets['train_data'] = datasets.ImageFolder(train_dir, transform=train_transforms)
image_datasets['valid_data'] = datasets.ImageFolder(valid_dir, transform=valid_transforms)
image_datasets['test_data'] = datasets.ImageFolder(test_dir, transform=test_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64)
testloader = torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64)
