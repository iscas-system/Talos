import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import timeit

def raw_execute_model(module, img_data, input_name="data", timing_count = 10):
    dtype = "float32"
    module.set_input(input_name, img_data)
    module.run()
    output_shape = (1, 1000)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).asnumpy()
    timing_number = timing_count
    timing_repeat = timing_count
    unoptimized = (
        np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
        * 1000
        / timing_number
    )
    unoptimized = {
        "mean": np.mean(unoptimized),
        "median": np.median(unoptimized),
        "std": np.std(unoptimized),
    }
    print(unoptimized)
    return unoptimized