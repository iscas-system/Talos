import torch
import torchvision
model = torchvision.models.resnet18()
inp   = torch.zeros([64, 3, 7, 7])
for temp in model.children():
    print(temp)
# for temp in model.children():
#      print(temp.)
# trace, grad = torch.jit._get_trace_graph(model, inp)
# print(trace)