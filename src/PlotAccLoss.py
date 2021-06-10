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
datause = 0.01
# X轴
x = []
for i in range(num_rounds):
    x.append(i)

# 数据分配
dataallocation_00 = config.get("DataAllocation_00")
dataallocation_01 = config.get("DataAllocation_01")
dataallocation_02 = config.get("DataAllocation_02")
dataallocation_03 = config.get("DataAllocation_03")

# 实例化类dataallocation_00
FL_0 = IIDlearning.FLLearning()
FL_0.dataallocation = dataallocation_00
FL_0.epochs = epochs_4
FL_0.DataUse = datause 
FL_0.num_rounds = num_rounds  
y_acctest_0, y_losstest_0 = FL_0.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1},本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_00,epochs_4))
del FL_0   # 收回类


# 实例化类dataallocation_01
FL_1 = IIDlearning.FLLearning()
FL_1.dataallocation = dataallocation_01
FL_1.epochs = epochs_4
FL_1.DataUse = datause 
FL_1.num_rounds = num_rounds     
y_acctest_1, y_losstest_1 = FL_1.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1},本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_01,epochs_4))
del FL_1   # 收回类

# 实例化类dataallocation_2
FL_2 = IIDlearning.FLLearning()  
FL_2.dataallocation = dataallocation_02
FL_2.epochs = epochs_4
FL_2.DataUse = datause 
FL_2.num_rounds = num_rounds   
y_acctest_2, y_losstest_2 = FL_2.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_02,epochs_4))
del FL_2   # 收回类

# 实例化类dataallocation_3
FL_3 = IIDlearning.FLLearning() 
FL_3.dataallocation = dataallocation_03
FL_3.epochs = epochs_4
FL_3.DataUse = datause 
FL_3.num_rounds = num_rounds    
y_acctest_3, y_losstest_3 = FL_3.main()   # 得到acc和loss
print("环境中有{0}个client,数据分配为{1}，本地训练轮数为{2}时，训练完成....".format(clientnum,dataallocation_03,epochs_4))
del FL_3   # 收回类

plt.figure() # 新建一个窗口
# acc test
plt.title('acc test')
plt.plot(x, y_acctest_0, color='green', label="[1.0,0,0,0]", linewidth=2, marker='*')
plt.plot(x, y_acctest_1, color='red', label="[0.25,0.25,0.25,0.25]", linewidth=2, marker='o')
plt.plot(x, y_acctest_2,  color='skyblue', label="[0.1,0.1,0.1,0.7]", linewidth=2, marker='p')
plt.plot(x, y_acctest_3, color='blue', label="[0.3,0.3,0.3,0.1]", linewidth=2, marker='+')
plt.xlabel('iteration times')
plt.ylabel('test accuracy(cn=4,DU=0.01,lr=0.01,epochs=5)')
plt.legend()    # 画出线lable
plt.gcf()    # 创建网格
plt.savefig('./save/acc_test_new_3.jpg')    # 保存图片
plt.show()

# loss test
plt.figure() # 新建一个窗口
plt.title('loss test')
plt.plot(x, y_losstest_0, color='green', label='[1.0,0,0,0]', linewidth=2, marker='*')
plt.plot(x, y_losstest_1, color='red', label='[0.25,0.25,0.25,0.25]', linewidth=2, marker='o')
plt.plot(x, y_losstest_2,  color='skyblue', label='[0.1,0.1,0.1,0.7]', linewidth=2, marker='p')
plt.plot(x, y_losstest_3, color='blue', label='[0.3,0.3,0.3,0.1]', linewidth=2, marker='+')
plt.xlabel('iteration times')
plt.ylabel('test loss(cn=4,U=0.01,lr=0.01,epochs=5)')
plt.legend()    # 画出线lable
plt.gcf()    # 创建网格
plt.savefig('./save/loss_test_new_3.jpg')    # 保存图片
plt.show()
#########################################################################