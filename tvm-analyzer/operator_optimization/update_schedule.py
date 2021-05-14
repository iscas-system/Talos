import tvm
import tvm.testing
from tvm import te
import numpy as np

def update_by_parallel(schedule, variables, name, factor):
    if "myadd" in name:
        schedule[variables[2]].parallel(variables[2].op.axis[0])
    return schedule

def update_by_vectorization_on_cpu(schedule, variables, name, factor):
    if "myadd" in name:
        outer, inner = schedule[variables[2]].split(variables[2].op.axis[0], factor=factor)
        schedule[variables[2]].parallel(outer)
        schedule[variables[2]].vectorize(inner)
    return schedule

def update_by_parallel_gpu(schedule, variables, name, factor):
    if "myadd" in name:
        bx, tx = schedule[variables[2]].split(variables[2].op.axis[0], factor=factor)
        schedule[variables[2]].bind(bx, te.thread_axis("blockIdx.x"))
        schedule[variables[2]].bind(tx, te.thread_axis("threadIdx.x"))
    return schedule