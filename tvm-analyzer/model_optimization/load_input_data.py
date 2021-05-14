import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor

def get_single_image(img_url="https://s3.amazonaws.com/model-server/inputs/kitten.jpg", img_name="imagenet_cat.png", img_resize = (224, 224), img_transpose = (2, 0, 1), normal_mean = [0.485, 0.456, 0.406], normal_std = [0.229, 0.224, 0.225], normal_reshape = (3, 1, 1)):
    img_url = img_url
    img_path = download_testdata(img_url, img_name, module="data")
    # Resize it to 224x224
    resized_image = Image.open(img_path).resize(img_resize)
    img_data = np.asarray(resized_image).astype("float32")
    # Our input image is in HWC layout while ONNX expects CHW input, so convert the array
    img_data = np.transpose(img_data, img_transpose)
    # Normalize according to the ImageNet input specification
    imagenet_mean = np.array(normal_mean).reshape(normal_reshape)
    imagenet_stddev = np.array(normal_std).reshape(normal_reshape)
    norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev
    # Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
    img_data = np.expand_dims(norm_img_data, axis=0)
    # print(img_data)
    return img_data