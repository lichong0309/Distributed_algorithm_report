# alg_FL算法和FedAvg算法在NonIID情况下的比较图
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import IIDlearning
import NonIIDlearning_1
from configuration import config
import csv
import alg_FL

# 信任矩阵
TrustMatrix = [[0,1,1,1,1,1,1,1,1,1],
               [1,0,1,1,1,1,1,1,1,1],
               [1,1,0,1,1,1,1,1,1,1],
               [1,1,1,0,1,1,1,1,1,1],
               [1,1,1,1,0,1,1,1,1,1],
               [1,1,1,1,1,0,1,1,1,1],
               [1,1,1,1,1,1,0,1,1,1],
               [1,1,1,1,1,1,1,0,1,1],
               [1,1,1,1,1,1,1,1,0,1],
               [1,1,1,1,1,1,1,1,1,0]]


class_pc = 1 
############################
##################################################################
clientnum = config.get("ClientNum_1") # 获得ClientNum
num_rounds = config.get("num_rounds") # 获得服务器训练轮数
epochs_4 = config.get("epochs_2")     # 获得本地训练轮数
datause = 1.0

dataallocation_0 = config.get("DataAllocation_10")      # [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dataallocation_1 = config.get("DataAllocation_14")      # [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
dataallocation_2 = config.get("DataAllocation_13")      # [0.075, 0.125, 0.025, 0.075, 0.2, 0.16, 0.05, 0.14, 0.1, 0.05]
dataallocation_3 = config.get("DataAllocation_12")      # [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.55]

    
# X轴
x = []
for i in range(num_rounds):
    x.append(i)


# alg算法;dataallocation_2
alg = alg_FL.alg_FL()
alg.dataallocation = dataallocation_2
alg.num_rounds = num_rounds
alg.classes_pc = class_pc
alg.TrustMatrix = TrustMatrix
alg.epochs = epochs_4
y_acctest_0, y_losstest_0 = alg.main()
del alg

# 非独立同分布：dataallocation_2
NonIID_FL_3 = NonIIDlearning_1.FLlearningNonIID_1()
NonIID_FL_3.dataallocation = dataallocation_2
NonIID_FL_3.num_rounds = num_rounds
NonIID_FL_3.classes_pc = class_pc
NonIID_FL_3.epochs = epochs_4
y_acctest_3, y_losstest_3 = NonIID_FL_3.main()
del NonIID_FL_3

plt.figure() # 新建一个窗口
# acc test
plt.title('acc test')
plt.plot(x, y_acctest_0, color='green', label="alg_DA2", linewidth=1, marker='*')
# plt.plot(x, y_acctest_1, color='red', label="IID_DA1", linewidth=1, marker='o')
# plt.plot(x, y_acctest_2,  color='skyblue', label="NonIID_DA1", linewidth=1, marker='p')
plt.plot(x, y_acctest_3, color='blue', label="NonIID_DA2,class_pc=1", linewidth=1, marker='+')
# plt.plot(x, y_acctest_4, color='k', label="NonIID_DA3", linewidth=1, marker='.')
plt.xlabel('iteration times')
plt.ylabel('test accuracy')
plt.legend()    # 画出线lable
plt.gcf()       # 创建网格
plt.savefig('./save/cc_test_alg_DA1.jpg')    # 保存图片
plt.show()


# # 非独立同分布：dataallocation_1
# NonIID_FL_2 = NonIIDlearning_1.FLlearningNonIID_1(dataallocation_1)
# NonIID_FL_2.num_rounds = num_rounds
# NonIID_FL_2.class_pc = class_pc
# y_acctest_2, y_losstest_2 = NonIID_FL_2.main()
# # writercsv(0,clientnum,dataallocation_1,epochs_4,y_acctest_2)
# print("Non_IID训练：{0}数据分配的情况下,训练完成".format(dataallocation_1))
# del NonIID_FL_2

# # 非独立同分布：dataallocation_3
# NonIID_FL_4 = NonIIDlearning_1.FLlearningNonIID_1(dataallocation_3)
# NonIID_FL_4.num_rounds = num_rounds
# NonIID_FL_4.class_pc = class_pc
# y_acctest_4, y_losstest_4 = NonIID_FL_4.main()
# # writercsv(0,clientnum,dataallocation_3,epochs_4,y_acctest_4)
# print("Non_IID训练：{0}数据分配的情况下,训练完成".format(dataallocation_3))
# del NonIID_FL_4


# plt.figure() # 新建一个窗口
# # acc test
# plt.title('acc test')
# # plt.plot(x, y_acctest_0, color='green', label="alg_DA2", linewidth=1, marker='*')
# # plt.plot(x, y_acctest_1, color='red', label="IID_DA1", linewidth=1, marker='o')
# plt.plot(x, y_acctest_2,  color='skyblue', label="NonIID_DA1", linewidth=1, marker='p')
# plt.plot(x, y_acctest_3, color='blue', label="NonIID_DA2", linewidth=1, marker='+')
# plt.plot(x, y_acctest_4, color='k', label="NonIID_DA3", linewidth=1, marker='.')
# plt.xlabel('iteration times')
# plt.ylabel('test accuracy')
# plt.legend()    # 画出线lable
# plt.gcf()       # 创建网格
# plt.savefig('./save/cc_test_00000.jpg')    # 保存图片
# plt.show()

# plt.figure() # 新建一个窗口
# # acc test
# plt.title('acc test')
# plt.plot(x, y_acctest_0, color='green', label="alg_DA2", linewidth=1, marker='*')
# # plt.plot(x, y_acctest_1, color='red', label="IID_DA1", linewidth=1, marker='o')
# plt.plot(x, y_acctest_2,  color='skyblue', label="NonIID_DA1", linewidth=1, marker='p')
# plt.plot(x, y_acctest_3, color='blue', label="NonIID_DA2", linewidth=1, marker='+')
# plt.plot(x, y_acctest_4, color='k', label="NonIID_DA3", linewidth=1, marker='.')
# plt.xlabel('iteration times')
# plt.ylabel('test accuracy')
# plt.legend()    # 画出线lable
# plt.gcf()       # 创建网格
# plt.savefig('./save/cc_test_alg.jpg')    # 保存图片
# plt.show()
