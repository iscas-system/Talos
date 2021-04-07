import torch
import torch.fx
import os
import psutil
import time
import executiveEngine
from typing import Dict,Tuple
from torch.fx.node import Node

print(os.cpu_count())
print(torch.cuda.device_count())
# print(psutil.cpu_count())

# cpu = torch.device('cpu',0)
# cuda = torch.device('cuda',0)
# print(cpu)
# print(cuda)
# print(torch.get_num_threads())
# print(torch.get_num_interop_threads())

# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        # x = x.to("cpu:0")
        y = self.linear(x+self.param)
        # y = y.to("cuda:0")
        return y.clamp(min=0.0, max=1.0)

env : Dict[str, Node] = {}

def load_arg(a):
    return torch.fx.graph.map_arg(a, lambda n: env[n.name])

def timeSpot1(*args):
    print(time.time_ns())
    # result = args[0].target(*load_arg(args[0].args), **load_arg(node.kwargs))
    print(len(args))
    for i in args:
        print(i)
        print("end i")
    print(len(args))
    return args[0]

def timeSpot2(*args):
    print(time.time_ns())
    # result = args[0].target(*load_arg(args[0].args), **load_arg(node.kwargs))
    print(len(args))
    for i in args:
        print(i)
        print("end i")
    print(len(args))
    return args[1]

def transform(m: torch.nn.Module,
              tracer_class : type = torch.fx.Tracer) -> torch.nn.Module:
    graph : torch.fx.Graph = tracer_class().trace(m)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    # for node in graph.nodes:
    #     # need to avoid recursive errors.
    #     if node.op == 'call_module':
    #         preNode1 = graph.call_function(timeSpot1,args=node._prev.args)
    #         preNode2 = graph.call_function(timeSpot2,args=node._prev.args)
    #         # afterNode = graph.call_function(timeSpot)
    #         # print("good")
    #         # node.append(afterNode)
    #         # print(node._next.target)
    #         node._prev.prepend(preNode1)
    #         node._prev.prepend(preNode2)
    #         node._prev.args = (preNode1,preNode2)
            
            # node.append(afterNode)
            # The target attribute is the function
            # that call_function calls.
            # node.target = torch.mul
            # with graph.inserting_before(node):
            #     print(timeSpot(node.args))
                 
            #     node.args = torch.fx.graph.map_arg((newnode,)[0], lambda x: x)

    graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.
    return torch.fx.GraphModule(m, graph)

module = MyModule().to("cpu")
symbolic_traced : torch.fx.GraphModule = torch.fx.symbolic_trace(module)
# input = torch.randn(3, 4,device="cpu")
# ret = torch.jit.trace(module,input)
print(symbolic_traced.graph)
sp = executiveEngine.ShapeProp(symbolic_traced)
input2 = torch.randn(3, 4,device="cpu")
# module2 = transform(module)
output2 = symbolic_traced.forward(x=input2)
print("Forward result:")
print(output2) 
print("Graph result:")
print(sp.propagate(input2))
# for node in symbolic_traced.graph.nodes:
#     print(node.op)
# print("After add fx graph:\n")
module2 = transform(module)
# print(module2.graph)

# input = torch.randn(3, 4,device="cpu")
# output = module.forward(x=input)
# print(output)

# with torch.autograd.profiler.profile(use_cuda=False) as prof:

# prof.export_chrome_trace("/tmp/test1.json")
# # Symbolic tracing frontend - captures the semantics of the module
# # High-level intermediate representation (IR) - Graph representation
# print("good")
# print(ret.graph)

# print(symbolic_traced.code)