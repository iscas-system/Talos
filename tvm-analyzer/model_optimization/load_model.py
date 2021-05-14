import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor

def read_onnx_modle(type="master/vision/classification/resnet/model/",name="resnet50-v2-7.onnx"):
    model_url = "".join(
        [
            "https://github.com/onnx/models/raw/",
            type,
            name,
        ]
    )
    model_path = download_testdata(model_url, name, module="onnx")
    onnx_model = onnx.load(model_path)
    return onnx_model