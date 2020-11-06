import torch
import torch.nn as nn
import os
from sklearn.datasets import load_boston

print(os.sys.path)

class LinearModel(nn.Module):
    def __init__(self, ndim):
        super(LinearModel,self).__init__()
        self.ndim = ndim
        self.weight = nn.Parameter(torch.randn(ndim,1))
        self.bias = nn.Parameter(torch.rand(1))

    def forward(self, x):
        x = torch.tensor(x,dtype=torch.float32)
        return x.mm(self.weight)+self.bias

boston = load_boston()

lm = LinearModel(13)
criterion = nn.MSELoss()
optimer = torch.optim.SGD(lm.parameters(),lr=1e-6)
data = torch.tensor(boston["data"],requires_grad=True)
target = torch.tensor(boston["target"],dtype=torch.float32)

for step in range(10000):
    predict = lm(data)
    loss = criterion(predict,target)
    if step % 100 == 0:
        print("Loss: {:.3f}".format(loss.item()))
    optimer.zero_grad()
    loss.backward()
    optimer.step()
