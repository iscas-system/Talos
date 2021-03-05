import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.markers as marker
import numpy as np
plt.figure(figsize=(8.5, 6))
plt.rcParams['font.family'] = ['Arial']
plt.tick_params(bottom='off',top='off',left = 'off',right='off')
ax = plt.axes()


ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


common_op_names = ['MaxPool2DWithIndicesBackward', 'NllLossBackward', 'ReluBackward1', 'ViewBackward', 'adaptive_avg_pool2d', 'add_', 'addcdiv_', 'addcmul_', 'any', 'clone', 'contiguous', 'conv2d', 'detach_', 'div', 'div_', 'empty', 'eq', 'flatten', 'is_complex', 'is_floating_point', 'is_nonzero', 'item', 'log_softmax', 'max_pool2d', 'mul_', 'nll_loss', 'ones_like', 'permute', 'random_', 'relu_', 'set_', 'size', 'slice', 'sqrt', 'stack', 'sub_', 'to', 'torch::autograd::AccumulateGrad', 'torch::autograd::GraphRoot', 'unsqueeze', 'view', 'zero_', 'zeros_like']


model_names = ["alexnet", "googlenet", "inceptionv3", "resnet50", "squeezenet", "vgg19"]


draw_op_names = ["conv2d", "max_pool2d", "div", "sqrt", "NllLossBackward"]
markers = ['o', 'v', '^', '<', '>', 's', 'x', 'D', 'd', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'None', None, ' ', '']
## 44 * 6
op_model_speedup = []
for op_name in common_op_names:
    model_speedup = []


    for model_name in model_names:
        csv_path = model_name + "_speedup_pytorch.csv"
        data = pd.read_csv(csv_path)
        
        model_speedup.append(data[data["operator"].isin([op_name])]["speedup"].values[0])
    op_model_speedup.append(model_speedup)


m = 0
for i, common_op_name in enumerate(common_op_names):
    if np.var(op_model_speedup[i]) > 5 :
        if m >= len(markers):
            m = 0
        plt.plot(model_names, op_model_speedup[i], label=common_op_name, color='k', linestyle="-", marker=markers[m], markersize=6)
        
        m = m + 1


# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8 , box.height])

plt.legend(loc='center left', bbox_to_anchor=(0.7, 0.9),ncol=1, frameon=False)
plt.xlabel("model", fontsize=12)
plt.ylabel("speedup ratio", fontsize=12)
plt.rcParams['font.family'] = ['Arial']
plt.savefig("compare.pdf")
plt.show()

