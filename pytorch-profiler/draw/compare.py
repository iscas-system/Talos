import pandas as pd

suffix = "_speedup_pytorch.csv"
names = ["alexnet", "bert", "googlenet", "inceptionv3", "resnet50", "squeezenet", "vgg19"]
n = len(names)

def round5(x):
    return round(x, 5)

def compare(model1, model2):
    ## params
    ## modelx[0]: operator names
    ## modelx[1]: speedup
    ## return
    ## compare[0]: operator names
    ## compare[1]: model1 speedup
    ## compare[2]: model2 speedup
    compare = [[], [], []]
    for i, name1 in enumerate(model1[0]):
        for j, name2 in enumerate(model2[0]):
            if name1 == name2:
                # compare.append([name1, model1[1][i], model2[1][j]])
                compare[0].append(name1)
                compare[1].append(model1[1][i])
                compare[2].append(model2[1][j])
    return compare

for i in range(n):
    for j in range(i + 1, n):
        model_name1 = names[i]
        model_name2 = names[j]
        data1 = pd.read_csv(model_name1 + suffix)
        data2 = pd.read_csv(model_name2 + suffix)

        model1 = [data1.iloc[:, 0].values, data1.iloc[:, 1].values]
        model2 = [data2.iloc[:, 0].values, data2.iloc[:, 1].values]
        result = compare(model1, model2)

        data_frame = pd.DataFrame({
            "operator_name" : result[0],
            model_name1 : map(round5, result[1]),
            model_name2 : map(round5, result[2]
        })

        same_count = len(result[0])
        path = "./compare/" + model_name1 + "_" + model_name2 + "_" + str(round(same_count / max(len(model1[0]), len(model2[0])), 2))
        data_frame.to_csv(path + ".csv", index=False)

        
