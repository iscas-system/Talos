import tvm
import tvm.testing
from tvm import te
import numpy as np
from describe_computation import create_simple_sum
from compile_operator import compile_execute_raw_operator
from execute_operator import evaluate_addition

# spefify a target:
tgt = tvm.target.Target(target="llvm", host="llvm")

# create schedulue an operator on the target:
schedules, variables = create_simple_sum()

# compile the operator
operator = compile_execute_raw_operator(schedules, variables, tgt, name="myadd")

# execute (compare, evaluate) operator by evaluate execution time:
evaluate_addition(operator, tgt, "naive", name="myadd")

# 
