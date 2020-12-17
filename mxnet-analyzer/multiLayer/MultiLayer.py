from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 理解全连接层的含义
W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))

print(W1)