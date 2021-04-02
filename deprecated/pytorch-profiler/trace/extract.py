import json
names = ["resnet50"]
def extract(name, c_start, c_end, g_start, g_end):
    
    f = open("./single/" + name + ".json", encoding="utf8")
    res = f.read()
    traces = json.loads(res)
    res = []
    for trace in traces:
        pid = trace["pid"]
        ts = trace["ts"]
        if pid == "CPU functions":
            if ts >= c_start and ts <= c_end:
                res.append(trace)

        if pid == "CUDA functions":
            if ts >= g_start and ts <= g_end:
                res.append(trace)
            
    with open("./epoch/" + name + "_epoch.json", "w+") as of:
        json.dump(res, of)
        print("done")
if __name__ == "__main__":

    # extract("alexnet", 1354992, 2514159, 816139, 1367422)
    # extract("googlenet", 2821622, 6234939, 1090056, 1856128)
    # extract("inceptionv3", 8533454, 17021914, 1749165, 3155914)
    # extract("resnet50", 6649676, 14794493, 1404457, 2499837)
    # extract("squeezenet", 1351623, 2717181, 989496, 1679126)
    # extract("vgg19", 15201773, 29725233, 1936519, 3459933)

    extract("bert", 6849670, 11239953, 1135300, 1578995)
