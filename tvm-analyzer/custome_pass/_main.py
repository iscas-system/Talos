import sys 
sys.path.append("..") 

import tvm
import tvm.testing
from tvm import te
import numpy as np

from operator_optimization import template_autotune,execute_autotune,compile_operator

schedule, args = template_autotune.create_matmul_parallel_template(1024,1024,1024,"float32")

# execute_autotune.execute_autotune_task("tutorial/matmul", "float32", 1024, 1024, 1024, tvm.target.Target(target="llvm", host="llvm"), 10, 5, "/root/temp/matmul.log")

matmul_module = compile_operator.compile_with_autune_log("/root/temp/matmul.log", schedule, args, tvm.target.Target(target="llvm", host="llvm"), "float32")

print(type(matmul_module))

@tvm.tir.transform.prim_func_pass(pass_func=matmul_module, opt_level=2, name="memory_plan")
def transform(func, mod, ctx):
    # my transformations here.
    print(type(func))
    return func

updated_mod = transform(matmul_module)