import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import json

plt.rcParams['font.family'] = ['Times New Roman']

fontsize = 14
plt.rcParams['font.size'] = fontsize


plt.figure(figsize=(6, 4))
bar_width = 1.0

plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.25)
# fig = plt.figure()
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
tick_label10000 = ["BERT\n + \nGoogleNet", "BERT\n + \nSqueezeNet", "ResNet-50\n + \nBERT", "AlexNet\n + \nSqueezeNet", "ResNet-50\n + \nGoogleNet", "GoogleNet\n + \nSqueezeNet", "BERT\n + \nAlexNet", "ResNet-50\n + \nSqueezeNet", "GoogleNet\n + \nAlexNet", "ResNet-50\n + \nAlexNet"]

tick_label5000 = ["ResNet-50\n + \nSqueezeNet", "GoogleNet\n + \nAlexNet", "ResNet-50\n + \nAlexNet"]
count_5000 = [27, 20, 21]

tick_label20000 = ["ResNet-50\n + \nSqueezeNet", "GoogleNet\n + \nAlexNet", "ResNet-50\n + \nAlexNet"]
count_20000 = [13, 15, 11]

tick_label03 = ["GoogleNet\n + \nAlexNet", "BERT\n + \nAlexNet", "ResNet-50\n + \nAlexNet"]
count_03 = [17, 7, 23]

tick_label07 = ["ResNet-50\n + \nSqueezeNet", "GoogleNet\n + \nAlexNet", "ResNet-50\n + \nAlexNet"]
count_07 = [19, 23, 17]

x = np.array([16,18,20])
# Threshold Line
# plt.plot([14] * 200, np.arange(200), color = "red", linestyle="--")
# Threshold Text
# plt.text(14.2, 150, "Affinity > 4")

plt.bar(x+bar_width, count_07, bar_width, color='#f5c89d', edgecolor='black', linewidth=0.6,label='Number of reassigned operations')
# plt.bar(x,y1_10000,bar_width,color='#f5c89d',label='Only GPU', edgecolor="black", linewidth=0.6)
# plt.bar(x+bar_width,y2_10000,bar_width,color='#aad4a3', label='UCGE+I', edgecolor="black", linewidth=0.6)
# plt.bar(x+2 * bar_width,y3_10000,bar_width,color='#fffeae', label='OSAS', edgecolor="black", linewidth=0.6)



plt.xticks(x + 1 * bar_width, tick_label07)
plt.ylim(0,30)
plt.xlabel("Task combination", fontweight ='bold')
plt.ylabel("Number of reassigned operations", fontweight ='bold')

# Add Exact Count
for i, c in enumerate(count_07):
    plt.text(x[i] + bar_width, count_07[i] + 0.5, str(count_07[i]), horizontalalignment="center")
plt.legend(ncol=3)
plt.savefig("move07.pdf")
plt.show()