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
    print(np.sum(gpu_tol_time) / 1000 / end_time)
    N = len(speedups)

    ax1 = plt.subplot(grid[i,j])
    plt.xlim(1e-2, 2 * 1e2)
    # plt.ylim(0)
    # plt.title(title)
    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax1.set_xscale("log")
    plt.xlabel("speedup rate", fontsize=fontsize)
    ax1.set_ylabel("CDF", fontsize=fontsize)

    X2 = np.sort(speedups)
    F2 = np.array(range(1, N + 1))/float(N)

    # ax1.plot(X2, F2, label="CDF")
    ax1.hist(X2, bins="auto")
    # hist, bin_edges = np.histogram(X2,bins=100)
    # print(hist)
    # print(bin_edges)
    # ax1.plot(hist)
    ax1.legend(loc='center left', bbox_to_anchor=(0.0, 0.9), frameon=False, fontsize=fontsize)

    line_x = [1] * 100
    line_y = np.linspace(0,1.0,100)
    ax1.plot(line_x, line_y, color="red")

    ax2 = ax1.twinx()
    # ax2.ticklabel_format(useOffset=False, style='plain')
    ax2.set_ylabel("runtime(%)", fontsize=fontsize)
    ax2.tick_params(axis="x", labelsize=fontsize)
    ax2.tick_params(axis="y", labelsize=fontsize)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))

    # ax2.plot(X2, gpu_tol_time, color="blue")
    # print(X2.shape)
    print(np.diff(X2))
    ax2.bar(X2, gpu_tol_time / np.array([10 * end_time]), width=0.1 * np.array(X2), color="orange", label="runtime(%)")
    # for i, x in enumerate(X2):
        # ax2.bar(x, gpu_tol_time[i] / 1000 / end_time * 100, width=0.2, color="orange", label="runtime(%)")
    # plt.legend(loc='center left', bbox_to_anchor=(0.7, 0.9),ncol=1, frameon=False)
    ax2.legend(["runtime"], loc='center left', bbox_to_anchor=(0.0, 0.8), frameon=False, fontsize=fontsize)


names = ["alexnet", "resnet50", "vgg19", "googlenet", "inceptionv3", "squeezenet", "bert"]
total_times = [7120.85874, 1.4477255614E4, 1.9326061621E4, 1.0063983269E4, 1.8342797123E4, 9062.36863, 6830.707]


for i, name in enumerate(names):
    draw(0, 0, name + "_speedup_pytorch.csv", name, total_times[i])
    plt.savefig(name + "_speedup_pytorch.pdf")
    plt.clf()

# draw(0, 0, "alexnet_speedup_pytorch.csv", "AlexNet", 7120.85874)
# draw(0, 0, "resnet50_speedup_pytorch.csv", "ResNet-50", 1.4477255614E4)
# draw(0, 0, "vgg19_speedup_pytorch.csv", "VGG-19", 1.9326061621E4)
# draw(0, 0, "googlenet_speedup_pytorch.csv", "GoogleNet", 1.0063983269E4)
# draw(0, 0, "inceptionv3_speedup_pytorch.csv", "Inception-v3", 1.8342797123E4)
# draw(0, 0, "squeezenet_speedup_pytorch.csv", "SqueezeNet", 9062.36863)
# draw(0, 0, "bert_speedup_pytorch.csv", "Bert", 6830.707)
# plt.savefig("alexnet_speedup_pytorch.pdf")
# plt.legend()
# plt.show()

# print(speedups)
# N = len(speedups)

# fig = plt.figure()

# plt.subplot(121)
# plt.title('BAR')
# hx, hy, _ = plt.hist(speedups, bins=50, color="lightblue")

# plt.ylim(0.0,max(hx)+0.05)
# plt.title('Generate random numbers \n from a standard normal distribution with python')
# plt.grid()

# plt.savefig("cumulative_density_distribution_01.png", bbox_inches='tight')

# plt.show()
# plt.close()

# plt.subplot(132)
# dx = hy[1] - hy[0]
# F1 = np.cumsum(hx)*dx

# plt.plot(hy[1:], F1)

# plt.title('CDF1')

# plt.savefig("cumulative_density_distribution_02.png", bbox_inches='tight')



# ax1 = plt.subplot(111)
# plt.title('CDF1')
# plt.xlabel("speedup")
# ax1.set_ylabel("cdf")
# X2 = np.sort(speedups)
# F2 = np.array(range(N))/float(N)

# ax1.plot(X2, F2)

# line_x = [1] * 100
# line_y = np.linspace(0,1.0,100)
# ax1.plot(line_x, line_y, color="red")


# ax2 = ax1.twinx()
# ax2.set_ylabel("GPU total time")
# ax2.plot(X2, gpu_tol_time, color="blue")
# for i, x in enumerate(X2):
#     # ax2.bar(x, cpu_tol_time[i], color="red")
#     ax2.plot(x, gpu_tol_time[i], color="blue")
#     print
# plt.subplot(133)


# x = np.linspace(0,80,N)
# y = rv_continuous.cdf(speedups)
# plt.ylim(0.0,1.0)
# plt.plot(x, y)
# plt.title('CDF2')

# plt.show()
# plt.close()



a = 1

class B:

    def __init__(self, a):
        self.a = a

    def f(self, c):
        a += 1
        g(c)


b = B()

b.f(1)