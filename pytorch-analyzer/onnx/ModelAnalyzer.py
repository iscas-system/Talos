import torch
from torchvision.models import resnet50

m = resnet50(pretrained=True)
static_model = torch.jit.trace(m,torch.randn(1,3,244,244))
print(static_model.forward.graph)
for children in static_model.children():
    print(children)