# 非独立同分布的情况下，是否参数聚合之前先在服务器中进行再训练
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import IIDlearning
from configuration import config 
from NonIIDlearning_2 import FLlearningNonIID_2
from dataallocationrandom import dataallocationrandom
from NonIIDlearning_1 import FLlearningNonIID_1

######################################
##############1、IID训练#################
clientnum = config.get("ClientNum_0") # 获得ClientNum
num_rounds = 100 # 获得服务器训练轮数
epochs_4 = config.get("epochs_4")     # 获得本地训练轮数
datause = 1.0

# 获取dataallocation数据分配
dataallocation1 = config.get("DataAllocation_11")           #  "DataAllocation_11":[0.09, 0.11, 0.09, 0.11, 0.09, 0.11, 0.09, 0.11, 0.09, 0.11]
dataallocation2 = config.get("DataAllocation_13")           #　　"DataAllocation_13":[0.075, 0.125, 0.025, 0.075, 0.2, 0.16, 0.05, 0.14, 0.1, 0.05]
# dataallocation1 = dataallocationrandom(10)
# dataallocation2 = dataallocationrandom(10)

# X轴
x = []
for i in range(num_rounds):
    x.append(i)

# # ①：独立同分布的数据训练情况下
# # 实例化类dataallocation_2
# FL_2 = learning.FLLearning(clientnum,dataallocation1,epochs_4,datause)  
# FL_2.num_rounds = num_rounds 
# y_acctest_2, y_losstest_2 = FL_2.main()   # 得到acc和loss
# print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation1,epochs_4))
# del FL_2   # 收回类

# ①：非独立同分布的情况下训练，②聚合之前不在服务器中进行在训练。
#############2、NonIID训练######################
print("\nNonIID训练：{}的数据分配情况，训练开始......".format(dataallocation1))
FL_non_0 = FLlearningNonIID_2()
FL_non_0.dataallocation = dataallocation1
FL_non_0.num_rounds = num_rounds
y_non_acc_test_0, y_non_losses_test_0 = FL_non_0.main()
print("NonIID训练：{}的数据分配情况，训练完成......".format(dataallocation1))
del FL_non_0

# ①：非独立同分布的情况下训练，②聚合之前在服务器中进行再训练。
print("\nNonIID训练：{}的数据分配情况，训练开始......".format(dataallocation1))
FL_non_1 = FLlearningNonIID_1()
FL_non_1.dataallocation = dataallocation1 
FL_non_1.num_rounds = num_rounds 
y_non_acc_test_1, y_non_losses_test_1 = FL_non_1.main()
print("NonIID训练：{}的数据分配情况,训练完成......".format(dataallocation1))
del FL_non_1


plt.figure() # 新建一个窗口
# acc test
plt.title('acc test')
# plt.plot(x, y_acctest_2,  color='skyblue', label="IID_DA1", linewidth=2, marker='p')
plt.plot(x, y_non_acc_test_0, color='green', label="NonIID_DA1", linewidth=2, marker='*')
plt.plot(x, y_non_acc_test_1, color='red', label="NonIID_DA2", linewidth=2, marker='o')
plt.xlabel('iteration times')
plt.ylabel('test accuracy')
plt.legend()    # 画出线lable
plt.gcf()    # 创建网格
plt.savefig('./save/acc_test_nonIID_2.jpg')    # 保存图片
plt.show()
