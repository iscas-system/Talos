import pandas as pd

suffix = "_speedup_pytorch.csv"
names = ["alexnet", "googlenet", "inceptionv3", "resnet50", "squeezenet", "vgg19"]
n = len(names)



same = []

sets = []

max_len = 0
for name in names:
    data = pd.read_csv(name + suffix)
    op_names = data.iloc[:, 0].values
    op_set = set(op_names)
    max_len = max(max_len, len(op_set))
    sets.append(op_set)


result = sets[0]
for i in range(1, len(sets)):
    result.intersection_update(sets[i])
print(len(result), len(result) / max_len)

print(sorted(result))