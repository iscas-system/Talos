import tvm
import tvm.testing
from tvm import te
import numpy as np

def compile_execute_raw_operator(schedule, variables, target, name="myadd"):
    fadd = tvm.build(schedules, variables, tgt, name= name)
    dev = tvm.device(target.kind.name, 0)
    if name == "myadd":
        n = 1024
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
    return fadd
