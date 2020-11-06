import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image, ImageFile
import os

a = torch.rand(10)
b = a.to(torch.device("cuda"))
print( b is a)

c = b.to(torch.device("cuda"))
print( b is c)

d = c.to(torch.device("cpu"))
print( d is c)