import tvm
import tvm.testing
from tvm import te
import numpy as np
from describe_computation import create_simple_sum
from compile_operator import compile_execute_raw_operator
from execute_operator import evaluate_addition
from update_schedule import update_by_parallel, update_by_vectorization_on_cpu

# spefify a target:
tgt = tvm.target.Target(target="llvm", host="llvm")
print (tgt.kind.name)

# create schedulue an operator on the target:
# schedules, variables = create_simple_sum()

# # compile the operator
# operator = compile_execute_raw_operator(schedules, variables, tgt, name="myadd")

# # execute (compare, evaluate) operator by evaluate execution time:
# evaluate_addition(operator, tgt, "naive", name="myadd")

# # update schedule:
# variables = update_by_parallel(variables)

# # recompile
# operator_parallel = compile_execute_raw_operator(schedules, variables, tgt, name="myadd_parallel")

# # execute
# evaluate_addition(operator_parallel, tgt, "parallel", name="myadd_parallel")