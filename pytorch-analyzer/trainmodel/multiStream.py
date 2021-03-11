import torch

# s1 = torch.cuda.Stream(device = 'cuda:0')
# s2 = torch.cuda.Stream(device = 'cuda:0')
# Initialise cuda tensors here. E.g.:
A = torch.rand(10000, 10000).cuda()
B = torch.rand(5000, 5000)
# Wait for the above tensors to initialise.
# torch.cuda.synchronize()

print(torch.__version__)
print(torch.cuda.is_available())
with torch.autograd.profiler.profile(use_cpu=True,use_cuda=True) as prof1:
    C = torch.mm(A, A)
    # with torch.cuda.stream(s2):
# with torch.autograd.profiler.profile(use_cpu=True,use_cuda=False) as prof2:
    D = torch.mm(B,B)
# with prof1:
#     E  = torch.add(A,A)
    # with torch.cuda.stream(s1):
    # with torch.cuda.stream(s2):
    #     D = torch.add(B,B)
    # Wait for C and D to be computed.
    # torch.cuda.synchronize()
    # Do stuff with C and D.
# print(type(prof.events()))
# print(prof.events().table( row_limit=-1))
# print(prof.activities)
prof1.export_chrome_trace("/tmp/both.json")
# prof2.export_chrome_trace("/tmp/only.json")