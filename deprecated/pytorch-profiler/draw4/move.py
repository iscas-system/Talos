import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import json

plt.rcParams['font.family'] = ['Times New Roman']

fontsize = 14
plt.rcParams['font.size'] = fontsize


plt.figure(figsize=(12, 4))
bar_width = 1.0
plt.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.24)

# fig = plt.figure()
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
tick_label10000 = ["BERT\n + \nGoogleNet", "BERT\n + \nSqueezeNet", "ResNet-50\n + \nBERT", "AlexNet\n + \nSqueezeNet", "ResNet-50\n + \nGoogleNet", "GoogleNet\n + \nSqueezeNet", "BERT\n + \nAlexNet", "ResNet-50\n + \nSqueezeNet", "GoogleNet\n + \nAlexNet", "ResNet-50\n + \nAlexNet"]

count_10000 = [38, 11, 169, 39, 103, 24, 9, 19, 21, 15]
# count_5000 = 
# count_20000 = 
# count_03 = 
# count_07 = 

x = 2 * np.arange(10)
# Threshold Line
plt.plot([14] * 200, np.arange(200), color = "red", linestyle="--")
# Threshold Text
plt.text(14.2, 150, "Affinity > 0.4")

plt.bar(x+bar_width, count_10000, bar_width, color='#f5c89d', edgecolor='black', linewidth=0.6,label='number of reassigned operations')
# plt.bar(x,y1_10000,bar_width,color='#f5c89d',label='Only GPU', edgecolor="black", linewidth=0.6)
# plt.bar(x+bar_width,y2_10000,bar_width,color='#aad4a3', label='UCGE+I', edgecolor="black", linewidth=0.6)
# plt.bar(x+2 * bar_width,y3_10000,bar_width,color='#fffeae', label='OSAS', edgecolor="black", linewidth=0.6)



plt.xticks(x + 1 * bar_width, tick_label10000)

plt.xlabel("Task combination", fontweight ='bold')
plt.ylabel("Number of reassigned operations", fontweight ='bold')

# Add Exact Count
for i, c in enumerate(count_10000):
    plt.text(x[i] + bar_width, count_10000[i] + 2, str(count_10000[i]), horizontalalignment="center")
plt.legend(ncol=3)
plt.savefig("move10000.pdf")
plt.show()