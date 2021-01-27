import json
import pandas as pd
names = ["alexnet", "googlenet", "inceptionv3", "resnet50", "squeezenet", "vgg19"]

def divide_by_count(name, trace_type, start, end, interval_num, phase_name, op_name):
    with open("./epoch_zero/" + name + "_epoch_zero.json", encoding="utf8") as f:
        res = f.read()
        traces = json.loads(res)

        res = []
        rec = []
        for trace in traces:
            pid = trace["pid"]
            ts = trace["ts"]
            if pid == trace_type and ts >= start - 1 and ts <= end + 1:
                res.append(trace)
                # rec.append(trace["name"])

        print("total" + str(len(res)))
        ops_per_interval = (len(res) + interval_num - 1) // interval_num

        current_count = 0
        start_time = -1
        index = 0
        for trace in res:
            if start_time == -1:
                start_time = trace["ts"]
            current_count += 1
            if current_count == ops_per_interval:
                print("Phase " + phase_name + " Inteval " + str(index) + ": time " + str(trace["ts"] + trace["dur"] - start_time))
                index += 1
                current_count = 0
                start_time = -1
        
        if(current_count > 0): 
            print("Phase " + phase_name + " Inteval " + str(index) + ": time " + str(end + res[-1]["dur"] - start_time))

def divide_by_speedup(name, trace_type, start, end):

    data = pd.read_csv("../" + name + "_speedup_pytorch.csv")
    speedup_records = []
    for row in data.iterrows():
        # print(row[1])
        speedup_records.append({"operator": row[1]["operator"], "speedup" : row[1][" speedup"]})
    # print(speedup_records)
    
    trace_file = open("./epoch_zero/" + name + "_epoch_zero.json", encoding="utf8")
    res = trace_file.read()
    traces = json.loads(res)

    speedup_interval = (-1, -1)
    index = 0
    start_time = -1
    count = 0
    total_count = 0
    time_series = []
    total_time = 0
    for trace in traces:
        name = trace["name"]
        pid = trace["pid"]
        ts = trace["ts"]
        speedup = -1
        for record in speedup_records:
            if name == record["operator"] or ('/' in record["operator"] and name in record["operator"]):
                speedup = record["speedup"]
                break
        if pid == trace_type:
            count += 1
            if start_time == -1:
                start_time = ts
            if speedup_interval == (-1, -1) and speedup != -1:
                # 存在speedup
                # if speedup < 1:
                #     speedup_interval = (speedup - 1, speedup + 5)
                # else :
                #     speedup_interval = (speedup * 0.1, speedup * 10)
                speedup_interval = (min(speedup - 1, speedup * 0.1), max(speedup + 5, speedup * 10))
                # print(speedup_interval)
            
            if speedup != -1 and (speedup < speedup_interval[0] or speedup > speedup_interval[1]):
                total_time += trace["ts"] - start_time
                print(" Inteval " + str(index) + ": time " + str(trace["ts"] - start_time) + " count " + str(count))
                time_series.append(trace["ts"] - start_time)
                # if speedup < 1:
                #     speedup_interval = (speedup - 1, speedup + 5)
                # else :
                #     speedup_interval = (speedup * 0.1, speedup * 10)
                speedup_interval = (min(speedup - 1, speedup * 0.1), max(speedup + 5, speedup * 10))


                start_time = trace["ts"]
                index += 1
                total_count += count
                count = 0
            

    print("total time:" + str(total_time))
    print("total count:" + str(total_count))

    return time_series
if __name__ == "__main__":

    # divide_by_count("alexnet", "CPU functions", 407667, 978806, 4, "forward", "")
    # divide_by_count("alexnet", "CUDA functions", 463985, 525239, 4, "forward", "")
    divide_by_speedup("alexnet", "CPU functions", 0, 0)
    divide_by_speedup("alexnet", "CUDA functions", 0, 0)
    # c = divide_by_speedup("googlenet", "CPU functions", 0, 0)
    # g = divide_by_speedup("googlenet", "CUDA functions", 0, 0)
    # print(len(c))
    # print(len(g))

    # for i in range(min(len(c), len(g))):
    #     print(c[i] / g[i])