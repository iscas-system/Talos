import matplotlib.pyplot as plt
import matplotlib as lib

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}

def drawChart(x, y, mk, lb):
    plt.xlabel('time(ms)', font)
    plt.ylabel('delay(m)', font)
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker=mk[i], label=lb[i])
    # 修改图例字体
    legend = plt.legend(prop=font1)
    # 修改刻度字体大小
    plt.tick_params(labelsize=12)
    # 修改标题
    plt.title("Performance of precision, recall and F1", font2)
    # 修改坐标轴标签
    plt.xlabel("bias parameter", font)
    plt.ylabel("value", font)
    plt.show()
    plt.close()
    return

# def drawSubChart(x, y, mk, lb, tl):
#     index = 221
#     for i in range(4):
#         plt.subplot(index+i)
#         for j in range(len(x[i])):
#             plt.plot(x[i][j], y[i][j], marker=mk[i][j], label=lb[i][j])
#             plt.title("Performance under different bias parameters", font2)
#             # 修改图例字体
#             legend = plt.legend(prop=font1)
#             # 修改刻度字体大小
#             plt.tick_params(labelsize=20)
#             # 修改坐标轴标签
#             plt.xlabel("round", font)
#             plt.ylabel("round", font)
#             # 修改标题
#             plt.title("Performance under different bias parameters", font2)
#     plt.subplots_adjust(hspace=0.5)
#     plt.show()
#     return

x = [[0,0.2,0.4,0.6,0.8,1.0],
     [0,0.2,0.4,0.6,0.8,1.0],
     [0,0.2,0.4,0.6,0.8,1.0]]
y = [[0.71,0.82,0.81,0.89,0.81,0.79],
     [0.65,0.512,0.74,0.76,0.75,0.70],
     [0.58,0.69,0.72,0.72,0.71,0.67]]
marker = ['o', 'x','|']
lb = ['Precision', 'Recall','F1']

x1 = [[[1,2,3,4,5]], [[1,2,3,4,5]], [[1,2,3,4,5]], [[1,2,3,4,5]]]
y1 = [[[2,1,5,3,4]], [[5,2,1,4,3]], [[5,2,1,4,3]], [[5,2,1,4,3]]]
marker1 = [['o'], ['x'], ['*'], ['+']]
lb1 = [['1'], ['2'], ['3'], ['4']]
title1 = [['1'], ['2'], ['3'], ['4']]

print(lib.matplotlib_fname())

drawChart(x, y, marker, lb)
# drawSubChart(x1, y1, marker1, lb1, title1)