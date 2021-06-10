# Plotepoch.py变量: epochs
# client数量：ClientNum_0: 4
# dataallocation : DataAllocation_02: [0.1,0.1,0.1,0.7]
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import IIDlearning
from configuration import config 

##################################################################
clientnum = config.get("ClientNum_0") # 获得ClientNum
num_rounds = config.get("num_rounds") # 获得服务器训练轮数
dataallocation_02 = config.get("DataAllocation_02")    # 训练的数据


# X轴
x = []
for i in range(num_rounds):
    x.append(i)

# epochs读取
epochs_0 = config.get("epochs_0")
epochs_1 = config.get("epochs_1")
epochs_2 = config.get("epochs_2")
epochs_3 = config.get("epochs_3")
epochs_4 = config.get("epochs_4")


# 实例化类dataallocation_00
FL_0 = IIDlearning.FLLearning() 
FL_0.dataallocation = dataallocation_02
FL_0.epochs = epochs_0 
FL_0.num_rounds = num_rounds   
y_acctest_0, y_losstest_0 = FL_0.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1},本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_02,epochs_0))
del FL_0   # 收回类

# 实例化类dataallocation_01
FL_1 = IIDlearning.FLLearning()
FL_1.dataallocation = dataallocation_02
FL_1.epochs = epochs_1 
FL_1.num_rounds = num_rounds    
y_acctest_1, y_losstest_1 = FL_1.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1},本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_02,epochs_1))
del FL_1   # 收回类


# 实例化类dataallocation_2
FL_2 = IIDlearning.FLLearning() 
FL_2.dataallocation = dataallocation_02
FL_2.epochs = epochs_2 
FL_2.num_rounds = num_rounds   
y_acctest_2, y_losstest_2 = FL_2.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_02,epochs_2))
del FL_2   # 收回类

# 实例化类dataallocation_3
FL_3 = IIDlearning.FLLearning()
FL_3.dataallocation = dataallocation_02
FL_3.epochs = epochs_3
FL_3.num_rounds = num_rounds    
y_acctest_3, y_losstest_3 = FL_3.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_02,epochs_3))
del FL_3   # 收回类

# 实例化类dataallocation_3
FL_4 = IIDlearning.FLLearning() 
FL_4.dataallocation = dataallocation_02
FL_4.epochs = epochs_4 
FL_4.num_rounds = num_rounds   
y_acctest_4, y_losstest_4 = FL_4.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_02,epochs_4))
del FL_4   # 收回类

# acc test
plt.figure()
plt.title('acc test(epochs)')
plt.plot(x, y_acctest_0, color='green', label="epochs=1", linewidth=2, marker='*')
plt.plot(x, y_acctest_1, color='red', label="epochs=2", linewidth=2, marker='o')
plt.plot(x, y_acctest_2,  color='skyblue', label="epochs=3", linewidth=2, marker='p')
plt.plot(x, y_acctest_3, color='blue', label="epochs=4", linewidth=2, marker='+')
plt.plot(x, y_acctest_4, color='k', label="epochs=5", linewidth=2, marker='.')
plt.xlabel('iteration times')
plt.ylabel('test accuracy')
plt.legend()    # 画出线lable
plt.gcf()    # 创建网格
plt.savefig('./save/acc_test_epochs.jpg')    # 保存图片
plt.show()

# loss test
plt.figure()
plt.title('loss test(epochs)')
plt.plot(x, y_losstest_0, color='green', label='epochs=1', linewidth=2, marker='*')
plt.plot(x, y_losstest_1, color='red', label='epochs=2', linewidth=2, marker='o')
plt.plot(x, y_losstest_2,  color='skyblue', label='epochs=3', linewidth=2, marker='p')
plt.plot(x, y_losstest_3, color='blue', label='epochs=4', linewidth=2, marker='+')
plt.plot(x, y_losstest_4, color='k', label="epochs=5", linewidth=2, marker='.')
plt.xlabel('iteration times')
plt.ylabel('test loss')
plt.legend()    # 画出线lable
plt.gcf()    # 创建网格
plt.savefig('./save/loss_test_epochs.jpg')    # 保存图片
plt.show()
#########################################################################