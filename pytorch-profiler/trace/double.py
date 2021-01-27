import json
import pandas as pd


names = ["alexnet", "squeezenet"]



def double(name):


    trace_file = open("./common/" + name + "_epoch.json", encoding="utf8")
    res = trace_file.read()
    traces = json.loads(res)
    
    output = []

    for trace in traces:

        trace["ts"] *= 8
        trace["dur"] *= 8


        output.append(trace)

    with open("./common/" + name + "x8_epoch.json", "w+") as of:
        json.dump(output, of)
        print("done")



if __name__ == "__main__":
    for name in names:
        double(name)