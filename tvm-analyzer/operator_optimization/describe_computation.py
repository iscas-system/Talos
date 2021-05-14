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

def create_matmul_basic(N,L,M, dtype):
    A = te.placeholder((N, L), name="a", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    # s = te.create_schedule(C.op)
    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    yo, yi = s[C].split(y, 8)
    xo, xi = s[C].split(x, 8)
    s[C].reorder(yo, xo, k, yi, xi)
    return None, [A, B, C]