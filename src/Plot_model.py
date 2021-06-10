# PlotAccLoss.py变量: dataallocation
# client数量：ClientNum_0: 4
# epochs: epochs_4: 5
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from configuration import config 

import modeltest_11_19
import modeltest_IID_11
import modeltest_11

##################################################################
clientnum = config.get("ClientNum_1") # 获得ClientNum
num_rounds = config.get("num_rounds") # 获得服务器训练轮数
epochs_4 = config.get("epochs_4")     # 获得本地训练轮数
dataallocation  = config.get("DataAllocation_14")                   # 数据分配
classes_pc = 2

# X轴
x = []
for i in range(num_rounds):
    x.append(i)

######modeltest_11_19##################
FL_0 = modeltest_11_19.FLNonIIDModel()
FL_0.num_rounds = num_rounds
FL_0.classes_pc = classes_pc
y_acc_test_0 , y_losses_test_0 = FL_0.main()
print("modeltest_11_19文件训练完成")
del FL_0


#########modeltest_11####################
FL_1 = modeltest_11.FLlearningNonIIDVGG11()
FL_1.num_rounds = num_rounds
FL_1.classes_pc = classes_pc
y_acc_test_1 , y_losses_test_1 = FL_1.main()
print("modeltest_11文件训练完成")
del FL_1

#########modeltest_IID_11 #############
FL_2 = modeltest_IID_11.FLLearningVGG11()
FL_2.num_rounds = num_rounds
y_acc_test_2 , y_losses_test_2 = FL_2.main()
print("modeltest_IID_11文件训练完成")
del FL_2

plt.figure() # 新建一个窗口
# acc test
plt.title('acc test')
plt.plot(x, y_acc_test_0, color='green', label="NonIID VGG11 VGG19", linewidth=1.5, marker='*')
plt.plot(x, y_acc_test_1, color='red', label="NonIID VGG11", linewidth=1.5, marker='o')
plt.plot(x, y_acc_test_2,  color='skyblue', label="IID VGG11", linewidth=1.5, marker='p')
plt.xlabel('iteration times')
plt.ylabel('test accuracy(cn=4,DU=0.01,lr=0.01,epochs=5)')
plt.legend()    # 画出线lable
plt.gcf()    # 创建网格
plt.savefig('./save/acc_model_test.jpg')    # 保存图片
plt.show()
