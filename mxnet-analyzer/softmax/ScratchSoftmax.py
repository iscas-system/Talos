# 定义超参数和参数，准备每个batch的训练数据，输入数据拉平
# 定义softmax的计算方法（准备softmax的数据）
# 定义线性模型，用softmax函数包装
# 交叉熵的定义与（1,0,0）及 （0.1,0.3,0.6）这几种方式做乘积，通过分类编号作为索引，直接计算（非索引的地方都是0），索引的y==1


from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()

y = np.array([0, 2])
y_hat = np.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
print(y_hat[[0, 1], y])