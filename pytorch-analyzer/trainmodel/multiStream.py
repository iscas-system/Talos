import torch

# s1 = torch.cuda.Stream(device = 'cuda:0')
# s2 = torch.cuda.Stream(device = 'cuda:0')
# Initialise cuda tensors here. E.g.:
A = torch.rand(10000, 10000, device = 'cuda:0')
B = torch.rand(5000, 5000,device = 'cuda:0')
# Wait for the above tensors to initialise.
# torch.cuda.synchronize()

with torch.profiler.profile() as prof:
    C = torch.mm(A, A)
    # with torch.cuda.stream(s2):
    D = torch.add(B,B)
    # with torch.cuda.stream(s1):
    # with torch.cuda.stream(s2):
    #     D = torch.add(B,B)
    # Wait for C and D to be computed.
    # torch.cuda.synchronize()
    # Do stuff with C and D.
print(prof.activities)
prof.export_chrome_trace("/workspace/allcuda.json")