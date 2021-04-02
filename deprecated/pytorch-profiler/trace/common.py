from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import json
import sys
sys.setrecursionlimit(1000000)

names = ["alexnet", "googlenet", "inceptionv3", "resnet50", "squeezenet", "vgg19"]
names = ["bert"]
def common(name):
    data = pd.read_csv("../" + name + "_speedup_pytorch.csv")
    speedup_records = []
    for row in data.iterrows():
        speedup_records.append({"operator": row[1]["operator"], "speedup" : row[1][" speedup"]})

    trace_file = open("./epoch_zero/" + name + "_epoch_zero.json", encoding="utf8")
    res = trace_file.read()
    traces = json.loads(res)

    cpu_traces = []
    gpu_traces = []
    total_time = 0
    for trace in traces:
        pid = trace["pid"]
        if pid == "CPU functions":
            cpu_traces.append(trace)
            total_time += trace["dur"]
        elif pid == "CUDA functions":
            gpu_traces.append(trace)
    

    # print(len(cpu_traces))
    # print(len(gpu_traces))
    # print(cpu_trace_names)
    # print(gpu_trace_names)

    cpu_indices = []
    gpu_indices = []
    p = min_delete(cpu_traces, gpu_traces)

    same(p, cpu_traces, gpu_traces, len(cpu_traces), len(gpu_traces), cpu_indices, gpu_indices)
    cpu_traces = np.array(cpu_traces)
    cpu_indices = np.array(cpu_indices)
    gpu_traces = np.array(gpu_traces)
    gpu_indices = np.array(gpu_indices)
    c_names = cpu_traces[cpu_indices]
    g_names = gpu_traces[gpu_indices]
    # print(len(c_names))
    # print(len(g_names))
    # for i in range(len(c_names)):
    #     if(c_names[i] != g_names[i]):
    #         print("false")
    #         print(c_names[i] + "," + g_names[i])
    # print("true")

    # for i in range(len(cpu_traces)):
    #     if cpu_traces[i]["name"] == "MkldnnConvolutionBackward":
    #         print("MkldnnConvolutionBackward in origin cpu_traces")

    # for i in range(len(cpu_traces[cpu_indices])):
    #     if cpu_traces[cpu_indices][i]["name"] == "MkldnnConvolutionBackward":
    #         print("MkldnnConvolutionBackward in filtered cpu_traces")

    res = np.concatenate((cpu_traces[cpu_indices], gpu_traces[gpu_indices])).tolist()

    i = 0
    j = 0
    diff = []
    sum_time = 0
    while i < len(cpu_traces) and j < len(cpu_traces[cpu_indices]):
        if cpu_traces[i] != cpu_traces[cpu_indices][j]:
            diff.append(cpu_traces[i])
            sum_time += cpu_traces[i]["dur"]
            i += 1
        else:
            i += 1
            j += 1
    # print(diff)
    print(sum_time / total_time)
        
    with open("./common/" + name + "_epoch.json", "w+") as of:
        json.dump(res, of)
        print("done")
    
def min_delete(a, b):
    m = len(a)
    n = len(b)
    dp = [[0 for i in range(n + 1)] for j in range(m + 1)]
    path = [["" for i in range(n + 1)] for j in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1]["name"] == b[j - 1]["name"] \
            or (a[i - 1]["name"] == "MkldnnConvolutionBackward" and b[j - 1]["name"] == "CudnnConvolutionBackward") \
            or (a[i - 1]["name"] == "NativeBatchNormBackward" and b[j - 1]["name"] == "CudnnBatchNormBackward"):
                dp[i][j] = dp[i - 1][j - 1] + a[i - 1]["dur"]
                # dp[i][j] = dp[i - 1][j - 1] + 1
                path[i][j] = "ul"
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                if dp[i - 1][j] > dp[i][j - 1]:
                    path[i][j] = "u"
                else :
                    path[i][j] = "l"
    return path

def same(path, a, b, i, j, a_res, b_res):
    if i == 0 or j == 0:
        return
    if path[i][j] == "ul":
        same(path, a, b, i - 1, j - 1, a_res, b_res)
        a_res.append(i - 1)
        b_res.append(j - 1)

    elif path[i][j] == "u":
        same(path, a, b, i - 1, j, a_res, b_res)
    else :
        same(path, a, b, i, j - 1, a_res, b_res)


if __name__ == "__main__":
    for name in names:
        common(name)


    # a = ["1","2","3","ConvolutionBackward","4","5","5","3","2","2","3"]
    # b = ["1","2","3","4","5","2","2","4","ConvolutionBackward"]

    # p = min_delete(a, b)
    # a_res = []
    # b_res = []
    # same(p, a, b, len(a), len(b), a_res, b_res)
    # a = np.array(a)
    # b = np.array(b)
    # print(a[a_res])
    # print(b[b_res])
