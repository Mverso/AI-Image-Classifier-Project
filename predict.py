import pandas as pd
import numpy as np

import torch
from torch import nn, optim

import torchvision
from torchvision import datasets, transforms, models

from PIL import Image
import PIL
from os import listdir
import json
import argparse

# Initiate variables with default values
checkpoint = 'checkpoint.pth'
filepath = 'cat_to_name.json'
arch=''
image_path = "flowers/test/10/image_07090.jpg"
topk = 5

# Set up parameters for entry in command line
parser = argparse.ArgumentParser()
parser.add_argument('-c','--checkpoint',
                    action='store',
                    type=str,
                    help='Name of file containing trained model to be loaded and used for predictions.')
parser.add_argument('-i','--image_path',
                    action='store',
                    type=str,
                    help='Location of image to predict')
parser.add_argument('-t', '--topk',
                    action='store',
                    type=int,
                    help='Choose quantity of classes visible for predictions.')
parser.add_argument('-j', '--json',
                    action='store',
                    type=str,
                    help='Name of json file with class names.')
parser.add_argument('-g','--gpu',
                    action='store_true',
                    help='Use GPU if available, if used CUDA in train.py, you must use it again.')

args = parser.parse_args()

# Select parameters entered in command line
if args.checkpoint:
    checkpoint = args.checkpoint
if args.image_path:
    image_path = args.image_path
if args.topk:
    topk = args.topk
if args.json:
    filepath = args.json
if args.gpu:
    device = 'cuda'


with open(filepath, 'r') as f:
    flower_to_name = json.load(f)

def load_checkpoint(filepath):
    '''
    load model from a checkpoint
    '''

    checkpoint = torch.load(filepath)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])


    return model

model = load_checkpoint(checkpoint)
if device == 'cuda':
    model.to('cuda')


def process_image(image):
    test_image = PIL.Image.open(image)

    # Get original dimensions
    width, height = test_image.size

    # Find shorter size and create settings to crop shortest side to 256
    if width < height: resize=[256, 10000]
    else: resize=[10000, 256]

    test_image.thumbnail(size=resize)

    # Find pixels to crop on to create 224x224 image
    width, height = test_image.size
    left = (width - 224)/2
    bottom = (height - 224)/2
    right = left + 224
    top = bottom + 224

    test_image = test_image.crop((left, bottom, right, top))

    test_image = np.array(test_image)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    test_image = (test_image - mean)/std

    # Move color channels to first dimension as expected by PyTorch
    test_image = test_image.transpose((2, 0, 1))

    return test_image
