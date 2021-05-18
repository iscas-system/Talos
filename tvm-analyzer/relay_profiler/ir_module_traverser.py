import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from memory_profiler import memory_usage

def profile_relay_operator(ir_module, ir_params):
    