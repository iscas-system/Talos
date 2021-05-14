from tvm.contrib import cc
from tvm.contrib import utils

def save_operator(operator, name):
    temp = utils.tempdir()
    fadd = operator
    fadd.save(temp.relpath(name+".o"))
    if tgt.kind.name == "cuda":
        fadd.imported_modules[0].save(temp.relpath(name+".ptx"))
    if tgt.kind.name == "rocm":
        fadd.imported_modules[0].save(temp.relpath(name+".hsaco"))
    if tgt.kind.name.startswith("opencl"):
        fadd.imported_modules[0].save(temp.relpath(name+".cl"))
    cc.create_shared(temp.relpath(name+".so"), [temp.relpath(name+".o")])
    print(temp.listdir())
    return temp.listdir()

def load_operator(listdir, name):
    fadd1 = tvm.runtime.load_module(temp.relpath(listdir[0]))
    fadd1_dev = tvm.target.Target(target="llvm", host="llvm")
    if tgt.kind.name == "cuda":
        fadd1_dev = tvm.runtime.load_module(temp.relpath(name+".ptx"))
        fadd1.import_module(fadd1_dev)

    if tgt.kind.name == "rocm":
        fadd1_dev = tvm.runtime.load_module(temp.relpath(name+".hsaco"))
        fadd1.import_module(fadd1_dev)

    if tgt.kind.name.startswith("opencl"):
        fadd1_dev = tvm.runtime.load_module(temp.relpath(name+".cl"))
        fadd1.import_module(fadd1_dev)
    
    return fadd1, fadd1_dev
    # fadd1(a, b, c)
    # tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())