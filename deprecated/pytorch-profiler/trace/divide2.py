import pandas as pd
import numpy as np
import json


from plot import plot
bot = 0.7
top = 1.42

name_and_speedup = []

def divide_by_speedup(model_name, trace_type, start, end):

    data = pd.read_csv("../" + model_name + "_speedup_pytorch.csv")
    speedup_records = []
    for row in data.iterrows():
        # print(row[1])
        speedup_records.append({"operator": row[1]["operator"], "speedup" : row[1][" speedup"]})
    # print(speedup_records)
    
    trace_file = open("./common/" + model_name + "_epoch.json", encoding="utf8")
    res = trace_file.read()
    traces = json.loads(res)
    speedup_interval = (-1, -1)

    split_indices = []
    split_times = []
    start_time = -1


    cpu_traces = []
    gpu_traces = []
    for i, trace in enumerate(traces):
        name = trace["name"]
        pid = trace["pid"]
        ts = trace["ts"]

        speedup = -1
        for record in speedup_records:
            if name == record["operator"] or ('/' in record["operator"] and name in record["operator"]):
                speedup = record["speedup"]
                break

        # 找到了speedup
        if pid == "CPU functions":
            cpu_traces.append(trace)
            if start_time == -1:
                
                start_time = trace["ts"]
                # print(start_time)
            # print(speedup)
            if speedup_interval == (-1, -1):
                speedup_interval = (min(speedup - 1, speedup * bot), max(speedup + 4, speedup * top))
                # speedup_interval = (speedup * bot, speedup * top)
            if speedup > 0.2 and (speedup < speedup_interval[0] or speedup > speedup_interval[1]) and ts - start_time >= 10000:
                # print(speedup, name)
                split_indices.append(i)
                split_times.append(ts - start_time)
                start_time = ts
                speedup_interval = (min(speedup - 1, speedup * bot), max(speedup + 1, speedup * top))
                # speedup_interval = (speedup * bot, speedup * top)
                # print(speedup_interval)
        if pid == "CUDA functions":
            gpu_traces.append(trace)
    print(split_indices)
    # print(split_times)
    # print(np.sum(split_times))
    split_times.append(cpu_traces[-1]["ts"] + cpu_traces[-1]["dur"] - start_time)

    print(len(split_indices), len(split_times))
    gpu_split_times = []
    start_time = 0

    j = 0
    
    for i, trace in enumerate(gpu_traces):
        ts = trace["ts"]

        if j < len(split_indices) and i == split_indices[j] :
            gpu_split_times.append(ts - start_time)
            start_time = ts
            j += 1
    gpu_split_times.append(gpu_traces[-1]["ts"] + gpu_traces[-1]["dur"] - start_time)
    # print(np.sum(split_times[1:]))
    # print(np.sum(gpu_split_times[1:]))
    # print(split_times[1:][3:])
    # print(np.sum(gpu_split_times[1:][3:]))

    print(model_name)
    print(split_times)
    print(gpu_split_times)
    print(np.around(np.array(split_times) / np.array(gpu_split_times), decimals=2))
    print("__________________________________")
    # plot(model_name, split_indices)
    if model_name == "bert" :
        name_and_speedup.append({"name" : model_name, \
        "speedup" : np.sum(split_times[2:]) / np.sum(gpu_split_times[2:]), \
        "cpu_time" : split_times[2:], \
        "gpu_time" : gpu_split_times[2:]})
    else :
        name_and_speedup.append({"name" : model_name, \
        "speedup" : np.sum(split_times[1:]) / np.sum(gpu_split_times[1:]), \
        "cpu_time" : split_times[1:], \
        "gpu_time" : gpu_split_times[1:]})
    return split_indices[1:]


    # speedup_interval = (-1, -1)
    # index = 0
    # start_time = -1
    # count = 0
    # total_count = 0
    # time_series = []
    # total_time = 0
    # for trace in traces:
    #     name = trace["name"]
    #     pid = trace["pid"]
    #     ts = trace["ts"]
    #     speedup = -1
    #     for record in speedup_records:
    #         if name == record["operator"] or ('/' in record["operator"] and name in record["operator"]):
    #             speedup = record["speedup"]
    #             break
    #     if pid == trace_type:
    #         count += 1
    #         if start_time == -1:
    #             start_time = ts
    #         if speedup_interval == (-1, -1) and speedup != -1:
    #             # 存在speedup
    #             # if speedup < 1:
    #             #     speedup_interval = (speedup - 1, speedup + 5)
    #             # else :
    #             #     speedup_interval = (speedup * 0.1, speedup * 10)
    #             speedup_interval = (min(speedup - 1, speedup * 0.1), max(speedup + 5, speedup * 10))
    #             # print(speedup_interval)
            
    #         if speedup != -1 and (speedup < speedup_interval[0] or speedup > speedup_interval[1]):
    #             total_time += trace["ts"] - start_time
    #             print(" Inteval " + str(index) + ": time " + str(trace["ts"] - start_time) + " count " + str(count))
    #             time_series.append(trace["ts"] - start_time)
    #             # if speedup < 1:
    #             #     speedup_interval = (speedup - 1, speedup + 5)
    #             # else :
    #             #     speedup_interval = (speedup * 0.1, speedup * 10)
    #             speedup_interval = (min(speedup - 1, speedup * 0.1), max(speedup + 5, speedup * 10))


    #             start_time = trace["ts"]
    #             index += 1
    #             total_count += count
    #             count = 0

    # print("total time:" + str(total_time))
    # print("total count:" + str(total_count))

    # return time_series

if __name__ == "__main__":

    divide_by_speedup("alexnet", "CPU functions", 0, 0)
    divide_by_speedup("googlenet", "CPU functions", 0, 0)
    # divide_by_speedup("inceptionv3", "CPU functions", 0, 0)
    divide_by_speedup("resnet50", "CPU functions", 0, 0)
    divide_by_speedup("squeezenet", "CPU functions", 0, 0)
    # divide_by_speedup("vgg19", "CPU functions", 0, 0)
    divide_by_speedup("bert", "CPU functions", 0, 0)
    print(sorted(name_and_speedup, key=lambda x: x["speedup"]))
    
    # print(np.sum([34685.658999999985, 10297.23999999999, 56478.840000000084, 35559.40699999989, 32904.36100000003, 26197.219999999972, 24296.91100000008, 12243.158999999985, 22724.962000000058, 40544.77799999993]))
    # print(np.sum([40382.513000000035, 21304.403999999864, 19083.597999999998, 24550.94299999997, 13655.780000000028, 16578.079000000143, 25052.267999999924, 18487.17299999995, 12725.523999999976, 13878.533999999985, 11819.529000000097]))
    # print(len([40382.513000000035, 21304.403999999864, 19083.597999999998, 24550.94299999997, 13655.780000000028, 16578.079000000143, 25052.267999999924, 18487.17299999995, 12725.523999999976, 13878.533999999985, 11819.529000000097]))