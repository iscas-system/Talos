import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.ticker as ticker


plt.rcParams['font.family'] = ['Arial']
plt.tick_params(bottom='off',top='off',left = 'off',right='off')
fontsize = 18
grid = plt.GridSpec(1, 1)
plt.figure(figsize=(8, 6.5))

def draw(i, j, csv_path, title, end_time):
    ax = plt.axes()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])

    data = pd.read_csv(csv_path)
    print(data)
    print(data.shape)
    speedups = data.iloc[:, 1].values
    cpu_tol_time = data.iloc[:, 2].values
    gpu_tol_time = data.iloc[:, 3].values
    cpu_avg_time = data.iloc[:, 4].values
    gpu_avg_time = data.iloc[:, 5].values

    ax = plt.subplot(grid[i,j])

    x = np.sort(speedups)
    x = np.log10(x)
    print(x)
    plt.ylim(0, 40)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("speedup ratio", fontsize=fontsize)
    plt.ylabel("runtime percentage(%)", fontsize=fontsize)
    ax.hist(x, bins=np.linspace(-2, 2, 9), weights=np.zeros_like(x) + 1. / x.size * 100, color="red")
    ax.set_xticklabels(["", r"$10^{-2}$", "", r"$10^{-1}$", "", r"$1$", "", r"$10^1$", "", r"$10^2$"], fontsize=fontsize)

names = ["alexnet", "resnet50", "vgg19", "googlenet", "inceptionv3", "squeezenet", "bert"]
total_times = [7120.85874, 1.4477255614E4, 1.9326061621E4, 1.0063983269E4, 1.8342797123E4, 9062.36863, 6830.707]

for i, name in enumerate(names):
    draw(0, 0, name + "_speedup_pytorch.csv", name, total_times[i])
    plt.savefig(name + "_log.same.pdf")
    plt.clf()

