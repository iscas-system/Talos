import torch
import torchvision.models as models

alexnet = models.alexnet(num_classes=1000, pretrained=True)

resnet50 = torch.hub.load('pytorch/vision', 'resnet50')