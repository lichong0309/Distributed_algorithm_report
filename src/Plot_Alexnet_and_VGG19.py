## 1. VGG19在Cifar10数据集NonIID下的精度.
## 2. AlexNet在Cifar10数据集NonIID下的精度.
## 3. VGG19在Cifar10数据集IID下的精度.
## 4. AlexNet在Cifar10数据集IID下的精度.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from configuration import config 

import NonIIDlearning_1                         # 1. VGG19在Cifar10数据集NonIID下的精度.
import AlexNet_Cifar10_NonIID           # 2. AlexNet在Cifar10数据集NonIID下的精度.
import IIDlearning # VGG19在Cifar10数据集IID下的精度.
import AlexNet_IID #  4. AlexNet在Cifar10数据集IID下的精度.


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

####  1. VGG19在Cifar10数据集NonIID下的精度.   ### 
FL_0 = NonIIDlearning_1.FLlearningNonIID_1()
FL_0.num_rounds = num_rounds 
FL_0.classes_pc = classes_pc
FL_0.dataallocation = dataallocation 
y_acc_test_0, y_losses_test_0 = FL_0.main()
print("VGG19在Cifar10数据集NonIID下的精度")
del FL_0

#### 2.  AlexNet在Cifar10数据集NonIID下的精度.    ####
FL_1 = AlexNet_Cifar10_NonIID.AlexNetFLlearningNonIID_1()
FL_1.num_rounds  = num_rounds
FL_1.classes_pc = classes_pc
FL_1.dataallocation = dataallocation
y_acc_test_1, y_losses_test_1 = FL_1.main()
print("AlexNet在Cifar10数据集NonIID下的精度")
del FL_1


### 3.VGG19在Cifar10数据集IID下的精度. ###### 
FL_2 = IIDlearning.FLLearning()
FL_2.num_rounds = num_rounds
FL_2.classes_pc = classes_pc
FL_2.dataallocation = dataallocation
y_acc_test_2, y_losses_test_2 = FL_2.main()
print("VGG19在Cifar10数据集IID下的精度")
del FL_2

##### 4. AlexNet在Cifar10数据集IID下的精度.  ########
FL_3 = AlexNet_IID.AlexNetFLLearning()
FL_3.num_rounds = num_rounds
FL_3.classes_pc = classes_pc
FL_3.dataallocation = dataallocation
y_acc_test_3, y_losses_test_3 = FL_3.main()
print(" AlexNet在Cifar10数据集IID下的精度")
del FL_3

plt.figure() # 新建一个窗口
# acc test
plt.title('acc test')
plt.plot(x, y_acc_test_0, color='green', label="NonIID VGG19", linewidth=1.5, marker='*')
plt.plot(x, y_acc_test_1, color='red', label="NonIID AlexNet", linewidth=1.5, marker='o')
plt.plot(x, y_acc_test_2,  color='skyblue', label="IID VGG19", linewidth=1.5, marker='p')
plt.plot(x, y_acc_test_3, color='blue', label='IID AlexNet', linewidth=1.5, marker='+')
plt.xlabel('iteration times')
plt.ylabel('test accuracy')
plt.legend()    # 画出线lable
plt.gcf()    # 创建网格
plt.savefig('./save/acc_model_1.jpg')    # 保存图片
plt.show()


plt.figure() # 新建一个窗口
# acc test
plt.title('acc test')
plt.plot(x, y_acc_test_0, color='green', label="NonIID VGG19", linewidth=1.5, marker='*')
plt.plot(x, y_acc_test_1, color='red', label="NonIID AlexNet", linewidth=1.5, marker='o')
plt.xlabel('iteration times')
plt.ylabel('test accuracy')
plt.legend()    # 画出线lable
plt.gcf()    # 创建网格
plt.savefig('./save/acc_model_2.jpg')    # 保存图片
plt.show()