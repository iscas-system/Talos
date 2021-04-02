import torch
import torch.fx
import os
import psutil
import time

# print(os.cpu_count())
# print(torch.cuda.device_count())
# print(psutil.cpu_count())

cpu = torch.device('cpu',0)
cuda = torch.device('cuda',0)
print(cpu)
print(cuda)
print(torch.get_num_threads())
print(torch.get_num_interop_threads())

# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        # x = x.to("cpu:0")
        xx = torch.add(x,self.param)
        y = self.linear(xx)
        y = y.to("cuda:0")
        return y.clamp(min=0.0, max=1.0)

def timeSpot():
    print(time.time_ns())
    return torch.rand(3, 4)

def transform(m: torch.nn.Module,
              tracer_class : type = torch.fx.Tracer) -> torch.nn.Module:
    graph : torch.fx.Graph = tracer_class().trace(m)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add
        if node.op == 'call_function':
            # The target attribute is the function
            # that call_function calls.
            if node.target == torch.add:
                # node.target = torch.mul
                with graph.inserting_after(node):
                    graph.call_function(timeSpot)
    graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.
    return torch.fx.GraphModule(m, graph)

module = MyModule().to(cpu)
symbolic_traced : torch.fx.GraphModule = torch.fx.symbolic_trace(module)
# input = torch.randn(3, 4,device="cpu")
# ret = torch.jit.trace(module,input)
print(symbolic_traced.graph)
module2 = transform(module)
print(module2.graph)

# input = torch.randn(3, 4,device="cpu")
# output = module.forward(x=input)
# print(output)

# input2 = torch.randn(3, 4,device="cpu")
# output2 = module2.forward(x=input)
# print(output2)

# with torch.autograd.profiler.profile(use_cuda=False) as prof:

# prof.export_chrome_trace("/tmp/test1.json")
# # Symbolic tracing frontend - captures the semantics of the module
# # High-level intermediate representation (IR) - Graph representation
# print("good")
# print(ret.graph)

# print(symbolic_traced.code)

