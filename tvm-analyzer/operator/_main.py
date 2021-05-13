import tvm
import tvm.testing
from tvm import te
import numpy as np
from describe_computation import create_simple_sum
from compile_operator import compile_execute_raw_operator
from execute_operator import evaluate_addition
from update_schedule import update_by_parallel, update_by_vectorization_on_cpu, update_by_parallel_gpu
from template_autotune import create_matmul_parallel_template
from execute_autotune import execute_autotune_task


def create_compile_execute_operator(create_fun, tgt, optimize_type="naive", name="myadd", update_schedule_fun=None, factor= 4):
    # create schedulue an operator without device:
    schedule, variables = create_fun()
    if update_schedule_fun != None:
        schedule = update_schedule_fun(schedule, variables, name, factor)
    # compile the operator on target:
    operator = compile_execute_raw_operator(schedule, variables, tgt, name=name)
    # execute (compare, evaluate) operator by evaluate execution time:
    evaluate_addition(operator, tgt, optimize_type, name, variables)
    return operator

# create_compile_execute_operator(create_simple_sum, tvm.target.Target(target="llvm", host="llvm"),name="myadd")

# create_compile_execute_operator(create_simple_sum, tvm.target.Target(target="llvm", host="llvm"), optimize_type="parallel", name="myadd_parallel", update_schedule_fun = update_by_parallel)

# create_compile_execute_operator(create_simple_sum, tvm.target.Target(target="llvm", host="llvm"), optimize_type="vectorization", name="myadd_vectorization", update_schedule_fun = update_by_vectorization_on_cpu, factor = 20)

# create_compile_execute_operator(create_simple_sum, tvm.target.Target(target="cuda", host="llvm"), optimize_type="parallel", name="myadd_cuda_parallel", update_schedule_fun = update_by_parallel_gpu, factor = 64)

shcedule,variables = create_matmul_parallel_template(1024, 1024, 1024, "float32")
execute_autotune_task("tutorial/matmul", "float32", 1024, 1024, 1024, tvm.target.Target(target="llvm", host="llvm"), 10, 5, "/root/temp/matmul.log")