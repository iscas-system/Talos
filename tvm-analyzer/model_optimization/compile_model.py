import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
# , config={"tir.add_lower_pass": [(1, transform)]}
def compile_raw_onnx_model(onnx_model, img_data, transform, target="llvm", input_name="data"):
    shape_dict = {input_name: img_data.shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    return module,params,target,mod

def compile_tuned_onnx_model(tuning_option, mod, params, transform, target="llvm"):
    with tvm.autotvm.apply_history_best(tuning_option["tuning_records"]):
        with tvm.transform.PassContext(opt_level=3:
            lib = relay.build(mod, target=target, params=params)
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    return module