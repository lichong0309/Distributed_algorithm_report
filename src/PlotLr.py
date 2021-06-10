# PlotAccLoss.py变量: dataallocation
# client数量：ClientNum_0: 4
# epochs: epochs_4: 5
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import IIDlearning
from configuration import config 

##################################################################
clientnum = config.get("ClientNum_0") # 获得ClientNum
num_rounds = config.get("num_rounds") # 获得服务器训练轮数
epochs_4 = config.get("epochs_4")     # 获得本地训练轮数
datause = 1.0
lr_0 = 0.1
lr_1 = 0.05
lr_2 = 0.02
lr_3 = 0.01
lr_4 = 0.005

# X轴
x = []
for i in range(num_rounds):
    x.append(i)

# 数据分配
# dataallocation_00 = config.get("DataAllocation_00")
dataallocation_01 = config.get("DataAllocation_01")
# dataallocation_02 = config.get("DataAllocation_02")
# dataallocation_03 = config.get("DataAllocation_03")

# 实例化类dataallocation_00
FL_0 = IIDlearning.FLLearning()
FL_0.dataallocation = dataallocation_01
FL_0.epochs = epochs_4 
FL_0.num_rounds = num_rounds
FL_0.DataUse = datause  
FL_0.lr = lr_0 
y_acctest_0, y_losstest_0 = FL_0.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1},本地训练轮数为{2},lr={3}时，训练完成....".format(clientnum,dataallocation_01,epochs_4,FL_0.lr))
del FL_0   # 收回类


# 实例化类dataallocation_01
FL_1 = IIDlearning.FLLearning()
FL_1.dataallocation = dataallocation_01
FL_1.epochs = epochs_4 
FL_1.num_rounds = num_rounds
FL_1.DataUse = datause   
FL_1.lr = lr_1  
y_acctest_1, y_losstest_1 = FL_1.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1},本地训练轮数为{2},lr={3}时，训练完成....".format(clientnum,dataallocation_01,epochs_4,FL_1.lr))
del FL_1   # 收回类

# 实例化类dataallocation_2
FL_2 = IIDlearning.FLLearning() 
FL_2.dataallocation = dataallocation_01
FL_2.epochs = epochs_4 
FL_2.num_rounds = num_rounds
FL_2.DataUse = datause 
FL_2.lr = lr_2  
y_acctest_2, y_losstest_2 = FL_2.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2},ls={3}时，训练完成....".format(clientnum,dataallocation_01,epochs_4,FL_2.lr))
del FL_2   # 收回类

# 实例化类dataallocation_3
FL_3 = IIDlearning.FLLearning() 
FL_3.dataallocation = dataallocation_01
FL_3.epochs = epochs_4 
FL_3.num_rounds = num_rounds
FL_3.DataUse = datause 
FL_3.lr = lr_3 
y_acctest_3, y_losstest_3 = FL_3.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2},lr={3}时，训练完成....".format(clientnum,dataallocation_01,epochs_4,FL_3.lr))
del FL_3   # 收回类

# 实例化类dataallocation_4
FL_4 = IIDlearning.FLLearning() 
FL_4.dataallocation = dataallocation_01
FL_4.epochs = epochs_4 
FL_4.num_rounds = num_rounds
FL_4.DataUse = datause 
FL_4.lr = lr_4 
y_acctest_4, y_losstest_4 = FL_4.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2},lr={3}时，训练完成....".format(clientnum,dataallocation_01,epochs_4,FL_4.lr))
del FL_4   # 收回类

plt.figure() # 新建一个窗口
# acc test
plt.title('acc test(lr)')
plt.plot(x, y_acctest_0, color='green', label="lr=0.1", linewidth=2, marker='*')
plt.plot(x, y_acctest_1, color='red', label="lr=0.05", linewidth=2, marker='o')
plt.plot(x, y_acctest_2,  color='skyblue', label="lr=0.02", linewidth=2, marker='p')
plt.plot(x, y_acctest_3, color='blue', label="lr=0.01", linewidth=2, marker='+')
plt.plot(x, y_acctest_4, color='k', label="lr=0.005", linewidth=2, marker='.')
plt.xlabel('iteration times')
plt.ylabel('test accuracy')
plt.legend()    # 画出线lable
plt.gcf()    # 创建网格
plt.savefig('./acc_test_lr.jpg')    # 保存图片
plt.show()

# loss test
plt.figure() # 新建一个窗口
plt.title('loss test(lr)')
plt.plot(x, y_losstest_0, color='green', label='lr=0.1', linewidth=2, marker='*')
plt.plot(x, y_losstest_1, color='red', label='lr=0.05', linewidth=2, marker='o')
plt.plot(x, y_losstest_2,  color='skyblue', label='lr=0.02', linewidth=2, marker='p')
plt.plot(x, y_losstest_3, color='blue', label='lr=0.01', linewidth=2, marker='+')
plt.plot(x, y_losstest_4, color='k', label="lr=0.005", linewidth=2, marker='.')
plt.xlabel('iteration times')
plt.ylabel('test loss')
plt.legend()    # 画出线lable
plt.gcf()    # 创建网格
plt.savefig('./loss_test_lr.jpg')    # 保存图片
plt.show()
#########################################################################