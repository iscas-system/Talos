# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 7}
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
font3 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 5}

data = pd.read_csv('../csv/cnn_ops_speedup.csv')
# print(data)
data = data.sort_values(by='speedup_ratio')

output_data = data[['operator_name', 'speedup_ratio']]
output_data.to_csv('../csv/sort_ops_speedup.csv', index=False)

name = data['operator_name']
speedup = data['speedup_ratio']
x = list(range(len(name)))

x1 = x[0:3]
name1 = name[0:3]
speedup1 = speedup[0:3]

x2 = x[3:24]
name2 = name[3:24]
speedup2 = speedup[3:24]

x3 = x[24:]
name3 = name[24:]
speedup3 = speedup[24:]

total_width, n = 0.8, 1
width = total_width / n
print(data)
plt.bar(x1, speedup1, width=width, tick_label=list(name1), fc='b')
plt.bar(x2, speedup2, width=width, tick_label=list(name2), fc='g')
plt.bar(x3, speedup3, width=width, tick_label=list(name3), fc='r')

index = np.array([i for i in range(28)])
plt.xticks(index, index + 1)

plt.tick_params(labelsize=10)

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(8)

plt.xlabel("operator_name", font2)
plt.ylabel("speedup_ratio", font2)

plt.title("SpeedupRatio - Operator", font1)
plt.show()



