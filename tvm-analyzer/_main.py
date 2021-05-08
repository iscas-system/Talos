from load_model import read_onnx_modle
from load_input_data import get_single_image
from compile_model import compile_onnx_model
from execute_model import raw_execute_model

# read onnx model from github
onnx_model = read_onnx_modle()
# load image for network
img_data = get_single_image()
# compile onnx_model
module = compile_onnx_model(onnx_model,img_data)
unoptimized = raw_execute_model(module, img_data)