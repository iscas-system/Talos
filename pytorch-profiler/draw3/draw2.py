import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import json


names = ["alexnet", "resnet50", "vgg19", "googlenet", "inceptionv3", "squeezenet", "bert"]

names = ["alexnet", "resnet50"]
all_speedup = [2.40, 6.05]
def draw(i, name):
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
    matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['font.family'] = ['Times New Roman']
    fontsize = 14
    plt.rcParams['font.size'] = fontsize
    
    plt.figure(figsize=(12, 4))
    bar_width = 0.6
    plt.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.3)
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    csv_path = name + "_speedup_pytorch2.csv"

    data = pd.read_csv(csv_path)
    # print(data)
    # print(data.shape)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    names = data.iloc[:, 0].values
    speedups = data.iloc[:, 1].values
    
    print(names)
    print(speedups)

    x = np.array([i for i in range(len(names))])
    print(x)

    # plt.bar(0, 1, 1)
    plt.bar(x, np.log10(speedups) + 2,bar_width,color='#f5c89d',label='OnlyGPU', edgecolor="black", linewidth=0.6)
    # plt.xticks(x, names, rotation=45, ha="left")

    # Speedup 1 line
    plt.plot([-bar_width/2, len(x)], [2,2], color='black', linewidth=1)
    plt.plot([-bar_width/2, len(x)], np.log10([all_speedup[i], all_speedup[i]]) + 2, color='red', linewidth=1)
    plt.yticks([0, 1, 2, 3, 4], [r"$10^{-2}$", r"$10^{-1}$", r"$1$", r"$10^{1}$", r"$10^{2}$"])
    ax.set_xticks(x)
    # ax.tick_params(axis=u'x', which=u'major',length=0)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    plt.xlabel("Operation", fontweight ='bold')
    plt.ylabel("Speedup ratio", fontweight ='bold')

    plt.xlim(-bar_width / 2, len(x))

    plt.grid(axis="y", ls='--')
    plt.savefig(name + "_v4.pdf")
    plt.show()




if __name__ == "__main__":
    # print(c)
    # get_actual_time()
    for i, name in enumerate(names):
        draw(i, name)
