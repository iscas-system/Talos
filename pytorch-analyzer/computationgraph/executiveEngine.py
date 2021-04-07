import torch
import torch.fx
from torch.fx.node import Node
import time
import csv
import codecs

from typing import Dict

class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
    
    def set_header(self):
        return ['算子名称', 'CPU时间', 'GPU时间', '加速比','参数数据量']
    

    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}
        csv_con = []
        headers = self.set_header()

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr
        
        def execute_node(node):
            t1 = 0
            t2 = 0
            if node.op == 'placeholder':
                t1 = time.time_ns()
                result = next(args_iter)
                t2 = time.time_ns()
                return result, t1, t2
            elif node.op == 'get_attr':
                t1 = time.time_ns()
                result = fetch_attr(node.target)
                t2 = time.time_ns()
                return result, t1, t2
            elif node.op == 'call_function':
                t1 = time.time_ns()
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
                t2 = time.time_ns()
                return result, t1, t2
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                t1 = time.time_ns()
                result = getattr(self_obj, node.target)(*args, **kwargs)
                t2 = time.time_ns()
                return result, t1, t2
            elif node.op == 'call_module':
                t1 = time.time_ns()
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
                t2 = time.time_ns()
                return result, t1, t2
            return t1, t2
        
        for node in self.graph.nodes:
            if node.op != 'output':
                (result, t1, t2) = execute_node(node)
            

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return load_arg(result)