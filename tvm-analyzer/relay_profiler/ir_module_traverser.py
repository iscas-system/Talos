import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from memory_profiler import memory_usage

str = ""
op_index = 0

class op_graph:
    def __init__(self):
        self.dictionary = {}
        self.var_nodes = []
        self.call_nodes = []
    
    def insert_op(self, id, op):
        self.dictionary[id] = op
        if op.type == "call":
            self.call_nodes.append(op)
        if op.type == "var":
            self.var_nodes.append(op)
    
    def find_starting_ops(self):
        result = []
        for temp_op in self.call_nodes:
            if_started = True 
            for key, value in temp_op.prior:
                if op.prior[key][1].type == "call":
                    if_started = False
                    break
            if if_started == True:
                result.append(temp_op)
        return result
        
class op_node:
    # args_index:{id,}
    def __init__(self, type, current_index, op):
        self.type = type
        self.id = name + "-" + str(current_index)
        self.name = op.name
        self.attrs = op.attrs
        self.op_instance = op
        self.next = {}
        self.prior = {}
        self.performance_data = None
    
    def set_next(self, next_op_node, next_op_index, args_index):
        next_op_id = next_op_node.name + "-" + str(next_op_index)
        self.next[next_op_id] = (args_index, next_op_node)
        op.prior[self.id] = (args_index, self)

def profile_relay_operator(ir_module, ir_params,i):
    print("current entrance function length:", len(ir_module.functions.items()))
    print(ir_module)
    entrance_tuple = ir_module.functions.items()[0]
    global_var = entrance_tuple[0]
    main_function = entrance_tuple[1]
    print_tree(i,main_function.body.op.name)
    # for each_param in main_function.params:
    #     print(type(each_param))
    # print(type(main_function))
    recursive_traverse_op(main_function.body.op.name, main_function.body.attrs, main_function.body.args)

    print("IR graph")
    print(str)

def print_tree(i, op_name):
    global str
    for temp in range(i):
        str+=" "
    str+=op_name
    str+="\n"

def recursive_traverse_op(name, attrs, args):
    # print(name,attrs)
    for each_arg in args:
        if isinstance(each_arg, tvm.relay.expr.Call):
            recursive_traverse_op(each_arg.op, each_arg.attrs, each_arg.args)
        if isinstance(each_arg,tvm.relay.expr.Var):
            # print("param Node:",each_arg)
