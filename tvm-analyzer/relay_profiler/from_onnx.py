
import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from memory_profiler import memory_usage
from tvm.relay.testing import check_grad, run_infer_type
from tvm.relay.transform import gradient
from ir_module_traverser import construct_op_graph, profile_memory

# python -m memory_profiler tvm-analyzer/relay_profiler/from_onnx.py

model_url = "".join(
    [
        "https://gist.github.com/zhreshold/",
        "bcda4716699ac97ea44f791c24310193/raw/",
        "93672b029103648953c4e5ad3ac3aadf346a4cdc/",
        "super_resolution_0.2.onnx",
    ]
)
model_path = download_testdata(model_url, "super_resolution.onnx", module="onnx")
onnx_model = onnx.load(model_path)

from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
img_y, img_cb, img_cr = img_ycbcr.split()
x = np.array(img_y)[np.newaxis, np.newaxis, :, :]

target = "llvm"

input_name = "1"
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

construct_op_graph(mod)
print(mod)
profile_memory(params, x)

# 基础组件是module，里面包含函数和参数、参数类型
# print(mod)
# print(dir(mod))
# # print(mod.type_definitions)
# print(mod.get_global_vars()[0])
# print(len(mod.get_global_type_vars()))
# 查看所有function的结构,一个模型只有一个function作为入口,一般没有全局变量：

# 查看所有的输入参数，通过命名方式决定对应的测试数据和模型参数（对应可变状态）：
# print(main_function.params)
# # 查看function的主体,作为根部节点：
# print(type(main_function.body))
# print(main_function.body)
# # 遍历节点（call Node的数据结构）：
# print("elements in call Node:")
# # 依赖的自身输入参数结果（倒序树）：
# print(main_function.body.args)
# # 自己的算子操作名称：
# print(main_function.body.op)
# # 自己的属性：
# print(main_function.body.attrs['newshape'])
# for k,v in mod.functions.items():
#     print(type(k))
#     print(v)

# 尝试截断最后一个算子的参数，重新编译运行，测量时间和内存开销：
# temp_x = np.random.rand(672,672,1,1)
# temp_params_map = {}
# temp_tvm_var = tvm.relay.var("1",shape=temp_x.shape)
# # temp_params_map['1'] = temp_tvm_var
# temp_params_list = []
# temp_params_list.append(temp_tvm_var)
# temp_body = tvm.relay.Call(main_function.body.op,temp_params_list,attrs=main_function.body.attrs)
# temp_function = tvm.relay.Function(temp_params_list, temp_body)
# # print(temp_function)
# temp_functions = {"GlobalVar": None, "main": temp_function}
# temp_ir_module = tvm.ir.IRModule(functions=temp_functions)
# print(temp_ir_module)

# print(temp_params_map)
######################################################################
# Execute on TVM
# ---------------------------------------------

# def start_profile():
#     return

# print(memory_usage(proc=start_profile))
# print("zero ir")

# dtype = "float32"
# print(type(intrp))
# def myfunc():
#     # may consume too many memory.
#     tvm_output = intrp.evaluate()(tvm.nd.array(temp_x.astype(dtype)), **temp_params_map).asnumpy()
#     print(tvm_output)

# # myfunc()
# print(memory_usage(proc=myfunc))
# print("first ir")

# determine base:
# dshape = (1,)
# base_input_x = np.random.rand(1)
# base_x = tvm.relay.var("x", shape = dshape)
# base_x = relay.add(base_x, relay.const(1, "float32"))
# base_function = tvm.relay.Function(relay.analysis.free_vars(base_x), base_x)
# base_functions = {"GlobalVar": None, "main": base_function}
# base_ir_module = tvm.ir.IRModule(functions=base_functions)
# print(base_ir_module)

# with tvm.transform.PassContext(opt_level=1):
#     base_intrp = relay.build_module.create_executor("graph", base_ir_module, tvm.cpu(0), target)

# def myfunc_base():
#     # may consume too many memory.
#     tvm_output_base = base_intrp.evaluate()(tvm.nd.array(base_input_x.astype(dtype)), **temp_params_map).asnumpy()
#     print(tvm_output_base)

# # myfunc()
# print(memory_usage(proc=myfunc_base))
# print("second ir")

# get gradient:
# dshape = (2,2)
# sigmoid_input_x = np.random.rand(2,2).astype("float32")
# sigmoid_x = tvm.relay.var("input", shape = dshape)
# sigmoid_output = tvm.relay.sigmoid(sigmoid_x)
# sigmoid_function = tvm.relay.Function([sigmoid_x],sigmoid_output)
# sigmoid_function = run_infer_type(sigmoid_function)
# bwd_func = run_infer_type(gradient(sigmoid_function))

# print("bwd_func)")
# print(bwd_func)

# tvm_output_grad = tvm.relay.create_executor(
#     mod=tvm.IRModule.from_expr(sigmoid_function),device = tvm.cpu(0), target = target
# ).evaluate()(tvm.nd.array(sigmoid_input_x.astype(dtype)), **temp_params_map).asnumpy()

# print("output_grad")
# print(tvm_output_grad)

# intrp = relay.create_executor(device = tvm.cpu(0), target = target)
# op_res, (op_grad, ) = intrp.evaluate(bwd_func)(sigmoid_input_x)

# print("output_grad2")
# print(op_res)
# print(op_grad)
# sigmoid_functions = {"GlobalVar": None, "main": sigmoid_function}
# sigmoid_ir_module = tvm.ir.IRModule(functions=sigmoid_functions)
# sigmoid_ir_module = tvm.relay.transform.InferType()(sigmoid_ir_module)

# sigmoid_ir_module['main'] = relay.transform.gradient(sigmoid_ir_module['main'],  mod=sigmoid_ir_module, mode='higher_order')
# sigmoid_ir_module = relay.transform.InferType()(sigmoid_ir_module)

# e = tvm.relay.create_executor("graph", sigmoid_ir_module, tvm.cpu(0), target).evaluate()
# with tvm.transform.PassContext(opt_level=1):
#     grad_intrp = relay.build_module.create_executor("graph", sigmoid_ir_module, tvm.cpu(0), target)

######################################################################
# Display results
# ---------------------------------------------
# We put input and output image neck to neck. The luminance channel, `Y` is the output
# from the model. The chroma channels `Cb` and `Cr` are resized to match with a simple
# bicubic algorithm. The image is then recombined and converted back to `RGB`.
# from matplotlib import pyplot as plt

# out_y = Image.fromarray(np.uint8((tvm_output[0, 0]).clip(0, 255)), mode="L")
# out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
# out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
# result = Image.merge("YCbCr", [out_y, out_cb, out_cr]).convert("RGB")
# canvas = np.full((672, 672 * 2, 3), 255)
# canvas[0:224, 0:224, :] = np.asarray(img)
# canvas[:, 672:, :] = np.asarray(result)
# plt.imshow(canvas.astype(np.uint8))
# plt.show()

######################################################################
# Notes
# ---------------------------------------------
# By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
# retains that dynamism upon import, and the compiler attemps to convert the model
# into a static shapes at compile time. If this fails, there may still be dynamic
# operations in the model. Not all TVM kernels currently support dynamic shapes,
# please file an issue on discuss.tvm.apache.org if you hit an error with dynamic kernels.
#
# This particular model was build using an older version of ONNX. During the import
# phase ONNX importer will run the ONNX verifier, which may throw a `Mismatched attribute type`
# warning. Because TVM supports a number of different ONNX versions, the Relay model
# will still be valid.
