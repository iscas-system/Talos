from load_model import read_onnx_modle
from load_input_data import get_single_image
from compile_model import compile_raw_onnx_model,compile_tuned_onnx_model
from execute_model import raw_execute_model
from tune_model import xgb_tune_module
import tvm
from tvm import te, auto_scheduler, runtime, topi
from tvm.auto_scheduler import _ffi_api
from tvm.topi.utils import get_const_tuple
from tvm.topi.sparse.utils import random_bsr_matrix

@tvm.tir.transform.prim_func_pass(opt_level=3)
def transform(func, mod, ctx):
    # my transformations here.
    print(tvm.tir.analysis.calculate_workspace_bytes(func))
    print(type(mod))
    print(type(ctx))
    return func


# read onnx model from github
onnx_model = read_onnx_modle()
# load image for network
img_data = get_single_image()
# compile and execute onnx_model
raw_module,params,target,mod = compile_raw_onnx_model(onnx_model,img_data, transform)
# unoptimized = raw_execute_model(raw_module, img_data)

# tune onnx_model with xgb
# tuning_option = xgb_tune_module(target, params, mod)
# #recompile and execute onnx_model
# new_module = compile_tuned_onnx_model(tuning_option, mod, params, transform,target="llvm")
# # rerun new module:
# optimized = raw_execute_model(new_module, img_data)
