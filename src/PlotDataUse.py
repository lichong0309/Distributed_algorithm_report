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

# X轴
x = []
for i in range(num_rounds):
    x.append(i)
# DataUse:
datause_1 = 1.0
datause_2 = 0.5
datause_3 = 0.2
datause_4 = 0.1
datause_5 = 0.05
datause_6 = 0.01 

# 数据分配
# dataallocation_00 = config.get("DataAllocation_00")
dataallocation_01 = config.get("DataAllocation_01")
# dataallocation_02 = config.get("DataAllocation_02")
# dataallocation_03 = config.get("DataAllocation_03")

# 实例化类dataallocation_00
FL_0 = IIDlearning.FLLearning()
FL_0.dataallocation = dataallocation_01
FL_0.epochs = epochs_4
FL_0.DataUse = datause_1 
FL_0.num_rounds = num_rounds    
y_acctest_0, y_losstest_0 = FL_0.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1},本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_01,epochs_4))
del FL_0   # 收回类


# 实例化类dataallocation_01
FL_1 = IIDlearning.FLLearning() 
FL_1.dataallocation = dataallocation_01
FL_1.epochs = epochs_4
FL_1.DataUse = datause_2 
FL_1.num_rounds = num_rounds  
y_acctest_1, y_losstest_1 = FL_1.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1},本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_01,epochs_4))
del FL_1   # 收回类

# 实例化类dataallocation_2
FL_2 = IIDlearning.FLLearning()
FL_2.dataallocation = dataallocation_01
FL_2.epochs = epochs_4
FL_2.DataUse = datause_3 
FL_2.num_rounds = num_rounds     
y_acctest_2, y_losstest_2 = FL_2.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_01,epochs_4))
del FL_2   # 收回类

# 实例化类dataallocation_3
FL_3 = IIDlearning.FLLearning() 
FL_3.dataallocation = dataallocation_01
FL_3.epochs = epochs_4
FL_3.DataUse = datause_4 
FL_3.num_rounds = num_rounds    
y_acctest_3, y_losstest_3 = FL_3.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_01,epochs_4))
del FL_3   # 收回类

# 实例化类dataallocation_4
FL_4 = IIDlearning.FLLearning()
FL_4.dataallocation = dataallocation_01
FL_4.epochs = epochs_4
FL_4.DataUse = datause_5 
FL_4.num_rounds = num_rounds     
y_acctest_4, y_losstest_4 = FL_4.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_01,epochs_4))
del FL_4   # 收回类

# 实例化类dataallocation_5
FL_5 = IIDlearning.FLLearning() 
FL_5.dataallocation = dataallocation_01
FL_5.epochs = epochs_4
FL_5.DataUse = datause_6 
FL_5.num_rounds = num_rounds   
y_acctest_5, y_losstest_5 = FL_5.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_01,epochs_4))
del FL_5   # 收回类


plt.figure() # 新建一个窗口
# acc test
plt.title('acc test(DataUse)')
plt.plot(x, y_acctest_0, color='green', label="DU=1.0", linewidth=2, marker='*')
plt.plot(x, y_acctest_1, color='red', label="DU=0.5", linewidth=2, marker='o')
plt.plot(x, y_acctest_2,  color='skyblue', label="DU=0.2", linewidth=2, marker='p')
plt.plot(x, y_acctest_3, color='blue', label="DU=0.1", linewidth=2, marker='+')
plt.plot(x, y_acctest_4, color='k', label="DU=0.05", linewidth=2, marker='.')
plt.plot(x, y_acctest_5, color='m', label="DU=0.01", linewidth=2, marker='v')
plt.xlabel('iteration times')
plt.ylabel('test accuracy')
plt.legend()    # 画出线lable
plt.gcf()       # 创建网格
plt.savefig('./save/cc_test_DataUse.jpg')    # 保存图片
plt.show()


# loss test
plt.figure() # 新建一个窗口
plt.title('loss test(DataUse)')
plt.plot(x, y_losstest_0, color='green', label="DU=1.0", linewidth=2, marker='*')
plt.plot(x, y_losstest_1, color='red', label="DU=0.5", linewidth=2, marker='o')
plt.plot(x, y_losstest_2,  color='skyblue', label="DU=0.2", linewidth=2, marker='p')
plt.plot(x, y_losstest_3, color='blue', label="DU=0.1", linewidth=2, marker='+')
plt.plot(x, y_losstest_4, color='k', label="DU=0.05", linewidth=2, marker='.')
plt.plot(x, y_losstest_5, color='m', label="DU=0.01", linewidth=2, marker='v')
plt.xlabel('iteration times')
plt.ylabel('test loss')
plt.legend()    # 画出线lable
plt.gcf()    # 创建网格
plt.savefig('./save/loss_test_DataUse.jpg')    # 保存图片
plt.show()
#########################################################################