import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import json

plt.rcParams['font.family'] = ['Arial']

fontsize = 16
plt.rcParams['font.size'] = fontsize
plt.figure(figsize=(7, 6))

names = ["alexnet", "resnet50", "vgg19", "googlenet", "inceptionv3", "squeezenet", "bert"]
total_times = [7120.85874, 1.4477255614E4, 1.9326061621E4, 1.0063983269E4, 1.8342797123E4, 9062.36863, 6830.707]
actual_times = [2181.769677520752, 9084.002900695801, 14209.016284332276, 4811.087987609863, 12548.20677722168, 3740.8614811401376, 6317.1004415283205]
plt.tick_params(bottom='off',top='off',left = 'off',right='off')

def draw(csv_path, total_time):
    data = pd.read_csv(csv_path)
    # print(data)
    # print(data.shape)
    speedups = data.iloc[:, 1].values
    cpu_tol_time = data.iloc[:, 2].values
    gpu_tol_time = data.iloc[:, 3].values
    # cpu_avg_time = data.iloc[:, 4].values
    # gpu_avg_time = data.iloc[:, 5].values
    print(np.sum(gpu_tol_time) / 1000 / total_time)
    
    n = len(speedups)

    op_num_per_interval = [n // 10] * 10
    for i in range(n % 10):
        op_num_per_interval[i] += 1
    # print(op_num_per_interval)

    speedup_groups = []
    time_groups = []
    cpu_time_groups = []
    gpu_time_groups = []

    speedup_group = []
    time_group = []
    cpu_time_group = []
    gpu_time_group = []

    index = 0
    for i in range(n):
        speedup_group.append(speedups[i])
        time_group.append(gpu_tol_time[i])
        cpu_time_group.append(cpu_tol_time[i])
        gpu_time_group.append(gpu_tol_time[i])

        if len(speedup_group) == op_num_per_interval[index]:
            speedup_groups.append(speedup_group)
            time_groups.append(time_group)
            cpu_time_groups.append(cpu_time_group)
            gpu_time_groups.append(gpu_time_group)
            speedup_group = []
            time_group = []
            cpu_time_group = []
            gpu_time_group = []
            
            index += 1
    speedup_mean = np.array(list(map(np.mean, speedup_groups)))
    # speedup_mean = np.array(list(map(np.sum, cpu_time_groups))) / np.array(list(map(np.sum, gpu_time_groups)))
    time_sum = np.array(list(map(np.sum, time_groups)))
    time_percent = time_sum / 1000 / total_time * 100
    print(speedup_mean)
    print(time_percent)

    x = np.array([0.1 * i for i in range(1, 11)])
    fig, ax1 = plt.subplots()
    ax1.tick_params(bottom=False,top=False,left=False,right=False)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    width = 0.04
    b1 = ax1.bar(x - width / 2, speedup_mean, width=width, label="speedup rate", color="#2c72b7", edgecolor="black", linewidth=0.6)
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + width / 2, time_percent, width=width, label="runtime(%)", color="#ca5b2e", edgecolor="black", linewidth=0.6)
    ax1.set_xlabel("CDF", fontsize=fontsize)

    ax1.set_ylabel("speedup rate", fontsize=fontsize)
    ax2.set_ylabel("runtime(%)", fontsize=fontsize)

    ax1.tick_params(axis="y", labelsize=fontsize)
    ax2.tick_params(axis="y", labelsize=fontsize)
    plt.legend(handles=[b1, b2], fontsize=fontsize)
    plt.savefig("./" + name + "_version_3_old.pdf")

def get_actual_time():
    times = []


    for name in names:
        trace_file = open("../trace/" + name + ".json", encoding="utf8")
        res = trace_file.read()
        traces = json.loads(res)
        time = 0
        for trace in traces:
            if trace["pid"] == "CUDA functions":
                time += trace["dur"]
        times.append(time / 1000)

    print(times)




if __name__ == "__main__":
    # print(c)
    # get_actual_time()
    for i, name in enumerate(names):
        draw(name + "_speedup_pytorch.csv", actual_times[i])

