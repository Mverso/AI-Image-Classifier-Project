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
