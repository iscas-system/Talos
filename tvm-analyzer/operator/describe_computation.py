import tvm
import tvm.testing
from tvm import te
import numpy as np


def create_simple_sum():
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = te.create_schedule(C.op)
    return s, [A,B,C]