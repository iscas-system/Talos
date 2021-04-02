import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import json

plt.rcParams['font.family'] = ['Times New Roman']

fontsize = 14
plt.rcParams['font.size'] = fontsize


plt.figure(figsize=(12, 4))
bar_width = 0.8
plt.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.24)
# fig = plt.figure()
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
wait_time = [6.581895000001416, 2.0642479999996723, 7.568017999994568, 14.5416660000009, 5.202541000005789, 1.9760940000000409, 4.9122379999998955, 8.866816999999806, 1.413719000000041, 3.7166919999998065]
imm_wait_time = [17.019998999999835, 31.428489000000294, 24.26721799999941, 4.438918000000063, 6.605544999999926, 8.257898999999975, 88.85914899999997 / 3, 11.164372000000206, 4.066912000000011, 13.812037999999943]


x = 2 * np.arange(10)

aff = [0.13, 0.21, 0.27, 0.13, 0.23, 0.15, 0.24, 0.3, 0.35, 0.37]
coff = [0.9, 0.9,  0.9,  0.8,  0.8,  0.65, 0.65, 0.65, 0.65,  0.65]
print(np.array(aff) / np.array(coff))

tick_label10000 = ["BERT\n + \nGoogleNet", "BERT\n + \nSqueezeNet", "ResNet-50\n + \nBERT", "AlexNet\n + \nSqueezeNet", "ResNet-50\n + \nGoogleNet", "GoogleNet\n + \nSqueezeNet", "BERT\n + \nAlexNet", "ResNet-50\n + \nSqueezeNet", "GoogleNet\n + \nAlexNet", "ResNet-50\n + \nAlexNet"]
y1_10000 = [67.5121546316392, 27.72844112102773, 66.8136719985465, 64.33272025610101, 57.27446895456253, 39.164575741041716, 22.084149566158025, 10.12940089534967, 35.26697039639559, 2.6332453012973023]
y2_10000 = [67.72413932988722, 30.014083726965975, 68.33091995835828, 64.51004175794498, 58.15842680933311, 42.169183859493316, 18.58142763130518, 16.457837010434936, 38.243263524706066, 7.475369248975978]
y3_10000 = [67.7290102876307, 30.030045919691756, 69.013692366909467, 65.524076672306737, 59.07164441645269, 43.677394304308788, 19.05846850267839, 20.580795503887673, 47.86470815390274, 30.59873069350377]
y3_10001 = [9.137990102876307, 6.730045919691756, 24.013692366909467, 28.724076672306737, 28.07164441645269, 25.677394304308788, 15.05846850267839, 14.580795503887673, 47.86470815390274, 30.59873069350377]

count_10000 = [38, 11, 169, 39, 103, 24, 9, 19, 21, 15]

# runtime = 

print((1 - np.array(y3_10000) / 100))



# Threshold Line
# plt.plot([13.4] * 100, np.arange(100), color = "red", linestyle="--")
# Threshold Text
# plt.text(13.5, 95, "Affinity > 0.4")


plt.bar(x, imm_wait_time, bar_width, color='#f5c89d', edgecolor="black", linewidth=0.6, label='TOOA+I')
plt.bar(x + bar_width, wait_time, bar_width, color='#aad4a3', edgecolor="black", linewidth=0.6, label='OSRAS')
# plt.bar(x + bar_width, waittime_10000, bar_width, bottom=(1 - np.array(y3_10000) / 100) * alltime_10000 - waittime_10000, color='#fffeae', edgecolor="black", linewidth=0.6)
# plt.bar(x,y1_10000,bar_width,color='#f5c89d',label='Only GPU', edgecolor="black", linewidth=0.6)
# plt.bar(x+bar_width,y2_10000,bar_width,color='#aad4a3', label='UCGE+I', edgecolor="black", linewidth=0.6)
# plt.bar(x+2 * bar_width,y3_10000,bar_width,color='#fffeae', label='OSAS', edgecolor="black", linewidth=0.6)
plt.xticks(x + 0.5 * bar_width,tick_label10000)
for i in range(len(wait_time)):
    plt.text(x[i], imm_wait_time[i] + 0.5, str(format(imm_wait_time[i], '.1f')), horizontalalignment="center")
    plt.text(x[i] + bar_width, wait_time[i] + 0.5, str(format(wait_time[i], '.1f')), horizontalalignment="center")

plt.xlabel("Task combination", fontweight ='bold')
plt.ylabel("Idle time(ms)", fontweight ='bold')

plt.legend(ncol=3)
plt.savefig("wait.pdf")
plt.show()


