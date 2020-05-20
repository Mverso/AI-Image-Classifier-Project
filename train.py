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



def create_model(arch='densenet121',hidden_units=320,learning_rate=0.003):
    '''
    Function builds model
    '''
    model =  getattr(models,arch)(pretrained=True)
    in_features = model.classifier.in_features
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False


    # Build classifier for model
    classifier = nn.Sequential(nn.Linear(in_features, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)

    #if device = 'cpu':
    model.to(device)
    #else:
        #model = model.to('cuda')

    return model, criterion, optimizer,  classifier

model, criterion, optimizer, classifier = create_model(arch, hidden_units, learning_rate)

def train_model(model, criterion, optimizer, epochs=3):
    '''
    Function that trains pretrained model and classifier on image dataset and validates.
    '''
    steps = 0
    running_loss = 0
    print_every = 5


    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            #if device = 'cpu':
            inputs, labels = inputs.to(device), labels.to(device)
            #else:
                #inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        #if device = 'cpu':
                        inputs, labels = inputs.to(device), labels.to(device)
                        #else:
                            #inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

    return model



trained_model = train_model(model, criterion, optimizer, epochs)


def test_model(model):
   # Do validation on the test set
    #if device = 'cpu':
    model.to(device)
    #else:
        #model.to('cuda')
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            #if device = 'cpu':
            inputs, labels = inputs.to(device), labels.to(device)
            #else:
                #inputs, labels = inputs.to('cuda'), labels.to('cuda')
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Test accuracy: {accuracy/len(testloader):.3f}")

test_model(trained_model)

def save_model(model_trained):
    '''
    Function saves the trained model architecture.
    '''
    model.class_to_idx = image_datasets['train_data'].class_to_idx
    save_dir = ''
    checkpoint = {'input_size': 1024,
              'output_size': 102,
              'arch': 'densenet121',
              'classifier': classifier,
              'epochs': epochs,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}

    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = 'checkpoint.pth'

    torch.save(checkpoint, save_dir)

save_model(trained_model)
