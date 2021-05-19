import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from memory_profiler import memory_usage

str = ""

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
            