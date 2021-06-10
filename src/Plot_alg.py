# alg_FL算法和FedAvg算法在NonIID情况下的比较图
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import IIDlearning
import NonIIDlearning_1
from configuration import config
import csv
import alg_FL
import alg



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

data_per_client = [[   0,  1336,   118,   278,   779,     4,   362,   693,   982,   377, ],
 [ 199,     0,  1176,   197,   241,   288,  2200,    23,   293,   126, ],
 [ 254,   411,     0,   267,  2064,   127,   908,     8,   226,   161, ],
 [ 230,    76,  1257,     0,  2094,    13,   236,    61,   285,   363, ],
 [ 441,   889,    18,   626,     0,   443,   385,   724,    66,    79, ],
 [ 298,   138,   176,    42,   319,     0,   813,  1315,     7,   610, ],
 [ 604,   304,   134,   746,   630,     2,  989,     0,     0,     0, ],
 [   0,   270,   226,   135,   274,   344,     0,  1946,   475,   424, ],
 [   0,   760,   174,  1542,   380,    26,   212,     0,   209,   924, ],
 [ 155,   187,     9,   343,   634,   523,   115,     0,  1655,  1276, ]]

# X轴
x = []
for i in range(num_rounds):
    x.append(i)

#################独立同分布##########################
FL_0 = IIDlearning.FLLearning()
FL_0.num_rounds = num_rounds
FL_0.epochs = epochs_4
y_acctest_0, y_losstest_0 = FL_0.main()
del FL_0


#####################传输之前的数据 训练##############
FL_1 = alg.FLlearningNonIID_1()
FL_1.data_per_client = data_per_client
FL_1.num_rounds = num_rounds
FL_1.epochs = epochs_4
y_acctest_1, y_losstest_1 = FL_1.main()
del FL_1


################传输之后的数据 训练################
FL_2 = alg.FLlearningNonIID_1()
FL_2.num_rounds = num_rounds
FL_2.epochs = epochs_4
y_acctest_2, y_losstest_2 = FL_2.main()
del FL_2



plt.figure() # 新建一个窗口
# acc test
plt.title('acc test')
plt.plot(x, y_acctest_0, color='green', label="IID", linewidth=1, marker='*')
plt.plot(x, y_acctest_1, color='red', label="NonIID_befor", linewidth=1, marker='o')
plt.plot(x, y_acctest_2,  color='skyblue', label="NonIID_after", linewidth=1, marker='p')

plt.xlabel('iteration times')
plt.ylabel('test accuracy')
plt.legend()    # 画出线lable
plt.gcf()       # 创建网格
plt.savefig('./save/cc_test_alg_new.jpg')    # 保存图片
plt.show()


