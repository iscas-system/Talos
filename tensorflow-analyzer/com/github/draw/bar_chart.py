# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 7}
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 25}
name_list = ['Monday', 'Tuesday', 'Friday', 'Sunday']
num_list = [1.5, 0.6, 7.8, 6]
num_list1 = [1, 2, 3, 1]
x = list(range(len(num_list)))
total_width, n = 0.8, 2
width = total_width / n

plt.bar(x, num_list, width=width, label='cpu', fc='y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='gpu', tick_label=name_list, fc='r')
# 修改图例字体
legend = plt.legend(prop=font1)
# 修改刻度的字体大小
plt.tick_params(labelsize=20)
# 修改坐标轴标签的字体大小
plt.xlabel("round", font)
plt.ylabel("round", font)
# 修改标题的字体
plt.title("test", font2)

plt.show()