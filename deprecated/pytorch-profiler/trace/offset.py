import json


names = ["alexnet", "googlenet", "inceptionv3", "resnet50", "squeezenet", "vgg19"]
names = ["bert"]

def offset(name):
    with open("./epoch/" + name + "_epoch.json", encoding="utf8") as f:
        res = f.read()
        traces = json.loads(res)
        output = []
        cpu_offset = 0
        gpu_offset = 0
        for trace in traces:
            pid = trace["pid"]
            ts = trace["ts"]
            if pid == "CPU functions":
                if cpu_offset == 0:
                    cpu_offset = ts
                trace["ts"] = trace["ts"] - cpu_offset

            if pid == "CUDA functions":
                if gpu_offset == 0:
                    gpu_offset = ts
                trace["ts"] = trace["ts"] - gpu_offset
                
            output.append(trace)
        with open("./epoch_zero/" + name + "_epoch_zero.json", "w+") as of:
            json.dump(output, of)
        print("done")
if __name__ == "__main__":
    for name in names:
        offset(name)