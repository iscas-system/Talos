import torch

s1 = torch.cuda.Stream(device = 'cuda:0')
s2 = torch.cuda.Stream(device = 'cuda:0')
# Initialise cuda tensors here. E.g.:
A = torch.rand(5000, 5000, device = 'cuda:0')
B = torch.rand(5000, 5000, device = 'cuda:0')
# Wait for the above tensors to initialise.
torch.cuda.synchronize()

with torch.autograd.profiler.profile(use_cuda=True) as prof:

    with torch.cuda.stream(s2):
        D = torch.add(B,B)
    with torch.cuda.stream(s1):
        C = torch.mm(A, A)
    # with torch.cuda.stream(s2):
    #     D = torch.add(B,B)
    # Wait for C and D to be computed.
    # torch.cuda.synchronize()
    # Do stuff with C and D.

prof.export_chrome_trace("/tmp/cudastreamparallel.json")