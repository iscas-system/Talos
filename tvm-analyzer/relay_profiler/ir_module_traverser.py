import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from memory_profiler import memory_usage

class op_graph:
    def __init__(self):
        self.dictionary = {}
        self.var_nodes = []
        self.call_nodes = []
    
    def insert_op(self, current_op_node):
        if self.find_if_exist(current_op_node) != None:
            return
        self.dictionary[current_op_node.id] = current_op_node
        if current_op_node.type == "call":
            self.call_nodes.append(current_op_node)
        if current_op_node.type == "var":
            self.var_nodes.append(current_op_node)
    
    def find_starting_ops(self):
        result = []
        for temp_op in self.call_nodes:
            if_started = True 
            for key in temp_op.prior.keys():
                if temp_op.prior[key][1].type == "call":
                    if_started = False
                    break
            if if_started == True:
                result.append(temp_op)
        return result

    def find_if_exist(self, temp_op):
        for key in self.dictionary.keys():
            if self.dictionary[key].op_instance is temp_op:
                return self.dictionary[key]
        return None

    def print_self(self):
        for start_op in self.find_starting_ops():
            start_op.print_self()

class op_node:
    # args_index:{id,}
    def __init__(self, type, current_index, op, attrs = None):
        self.type = type
        if self.type == "var":
            self.name = op.name_hint
        if self.type == "call":
            self.name = op.op.name
            self.attrs = attrs
        self.id = self.name + "-" + str(current_index)
        self.op_instance = op
        self.next = {}
        self.prior = {}
        self.performance_data = {}
    
    def set_next(self, next_op_node, next_op_index, args_index):
        next_op_id = next_op_node.name + "-" + str(next_op_index)
        self.next[next_op_id] = (args_index, next_op_node)
        next_op_node.prior[self.id] = (args_index, self)
    
    def print_self(self):
        print(self.id, self.attrs, self.next, self.prior)

op_index = 0
computation_graph = op_graph()

def construct_op_graph(ir_module, ir_params):
    global op_index, computation_graph
    print("current entrance function length:", len(ir_module.functions.items()))
    print(ir_module)
    entrance_tuple = ir_module.functions.items()[0]
    global_var = entrance_tuple[0]
    main_function = entrance_tuple[1]
    for each_param in main_function.params:
        temp_op_node = op_node("var", op_index, each_param, attrs = None)
        computation_graph.insert_op(temp_op_node)
        op_index+=1
    recursive_traverse_op(main_function.body.attrs, main_function.body.args, temp_op=main_function.body)

    computation_graph.print_self()


def recursive_traverse_op(attrs, args, temp_op=None):
    global op_index, computation_graph
    # print(name,attrs)
    args_index = 0
    next_op_node = op_node("call", op_index, temp_op, attrs = attrs)
    op_index+=1
    for each_arg in args:
        if isinstance(each_arg, tvm.relay.expr.Call):
            current_node_op = recursive_traverse_op(each_arg.attrs, each_arg.args, temp_op = each_arg)
            current_node_op.set_next(next_op_node, op_index-1, args_index)
            computation_graph.insert_op(current_node_op)
            args_index+=1
        if isinstance(each_arg,tvm.relay.expr.Var):
            current_node_op = computation_graph.find_if_exist(each_arg)
            if current_node_op == None:
                current_node_op = op_node("var", op_index, each_arg, attrs = None)
            current_node_op.set_next(next_op_node, op_index-1, args_index)
            computation_graph.insert_op(current_node_op)
            args_index+=1
    computation_graph.insert_op(next_op_node)
    return next_op_node

def profile_relay_operator():
    return 