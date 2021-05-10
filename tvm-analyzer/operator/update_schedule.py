import tvm
import tvm.testing
from tvm import te
import numpy as np

def update_by_parallel(variables, name):
    if name == "myadd":
        variables[2].parallel(C.op.axis[0])
    return variables

def update_by_vectorization_on_cpu(variables, name, factor):
    if name == "myadd":
        variables[2].parallel(C.op.axis[0])
        outer, inner = variables[2].split(C.op.axis[0], factor=factor)
        variables[2].parallel(outer)
        variables[2].vectorize(inner)