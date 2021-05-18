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
    entrance_tuple = ir_module.functions.items()[0]
    global_var = entrance_tuple[0]
    main_function = entrance_tuple[1]
    print(type(main_function.params))
    print_tree(i,main_function.body.op.name)
    for each_param in main_function.params:
        print(type(each_param))
    print("")
    for each_arg in main_function.body.args:
        print(type(each_arg))

    print("IR graph")
    print(str)

def print_tree(i, op_name):
    global str
    for temp in range(i):
        str+=" "
    str+=op_name
    str+="\n"