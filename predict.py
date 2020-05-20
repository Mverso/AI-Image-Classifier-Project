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
