# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 25}
name_list = ['iQA', 'QA-KG[4]', 'CAN for QA[5]', 'CKB[6]']

num_list_metric = [0.91, 0.79, 0.82, 0.85]
num_list_log = [0.85, 0.82, 0.81, 0.85]
num_list_trace = [0.88, 0.66, 0.70, 0.79]
num_list_we= [0.86, 0.86, 0.77, 0.89]
x = list(range(len(num_list_we)))
total_width, n = 0.8, 4
width = total_width / n

plt.bar(x, num_list_metric, width=width, label='Percision', fc='darkblue')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list_log, width=width, label='Recall', tick_label=name_list, fc='yellow')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list_trace, width=width, label='F1', tick_label=name_list, fc='green')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, num_list_we, width=width, label='Our Approach', tick_label=name_list, fc='crimson')
# 修改图例字体
legend = plt.legend(prop=font1, loc='upper right')
# 修改刻度的字体大小
plt.tick_params(labelsize=10)
# 修改坐标轴标签的字体大小
#plt.ylabel("round", font)
plt.xlabel("KG based approaches", font1)
# 修改标题的字体
#plt.title("test", font)
plt.show()