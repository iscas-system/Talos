import timeit
import tvm
import tvm.testing
from tvm import te
import numpy as np

def generate_input(name, dev):
    mean_time = 0
    if name == "myadd":
        n = 32768
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        evaluator = func.time_evaluator(func.entry_name, dev, number=10)
        mean_time = evaluator(a, b, c).mean
    return mean_time

def evaluate_addition(func, target, optimization, name):
    dev = tvm.device(target.kind.name, 0)
    print("%s: %f" % (optimization, generate_input(name, dev)))