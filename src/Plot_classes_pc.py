# 给client分配不同的class时的训练精度比较图
from os import write
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import IIDlearning
import NonIIDlearning_1
from configuration import config
import csv

# 创建.csv文件函数
def creat_csv():
    path = "./datacsv/class.csv"
    with open(path,"wb") as f:
        csv_write = csv.writer(f)
        csv_head = ["classes in each client"]
        csv_write.writerow(csv_head)

# 写入文件函数
def write_data(data=[0]):
    path = "./datacsv/class.csv"
    with open(path,"a+") as f:
        csv_wite = csv.writer(f)
        data_row = data
        csv_wite.writerow(data_row)


# 创建文件
creat_csv()
############################
##################################################################
clientnum = config.get("ClientNum_1") # 获得ClientNum
num_rounds = 200 # 获得服务器训练轮数
epochs_4 = config.get("epochs_4")     # 获得本地训练轮数
datause = 1.0

dataallocation_1 = config.get("DataAllocation_14")      # [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


# classes_pc
classes_pc_1 = 1
classes_pc_2 = 2 
classes_pc_3 = 3 
# classes_pc_4 = 4
classes_pc_5 = 5
# classes_pc_6 = 6 
classes_pc_7 = 7
# classes_pc_8 = 8
classes_pc_9 = 9 



    
# X轴
x = []
for i in range(num_rounds):
    x.append(i)


# 非独立同分布：dataallocation_1,classes_pc_1 
NonIID_FL_1 = NonIIDlearning_1.FLlearningNonIID_1()
NonIID_FL_1.dataallocation = dataallocation_1
NonIID_FL_1.num_rounds = num_rounds
NonIID_FL_1.classes_pc = classes_pc_1
y_acctest_1, y_losstest_1 = NonIID_FL_1.main()
print("Non_IID训练：{0}数据分配的情况下,训练完成".format(dataallocation_1))
NonIID_FL_1.csvnum = 1
write_data(y_acctest_1)
del NonIID_FL_1

# 非独立同分布：dataallocation_1,classes_pc_2
NonIID_FL_2 = NonIIDlearning_1.FLlearningNonIID_1()
NonIID_FL_2.dataallocation = dataallocation_1
NonIID_FL_2.num_rounds = num_rounds
NonIID_FL_2.classes_pc = classes_pc_2
y_acctest_2, y_losstest_2 = NonIID_FL_2.main()
print("Non_IID训练：{0}数据分配的情况下,训练完成".format(dataallocation_1))
NonIID_FL_2.csvnum = 2
write_data(y_acctest_2)
del NonIID_FL_2

# 非独立同分布：dataallocation_1,classes_pc_3
NonIID_FL_3 = NonIIDlearning_1.FLlearningNonIID_1()
NonIID_FL_3.dataallocation = dataallocation_1
NonIID_FL_3.num_rounds = num_rounds
NonIID_FL_3.classes_pc = classes_pc_3
y_acctest_3, y_losstest_3 = NonIID_FL_3.main()
print("Non_IID训练：{0}数据分配的情况下,训练完成".format(dataallocation_1))
NonIID_FL_3.csvnum = 3
write_data(y_acctest_3)
del NonIID_FL_3

# 非独立同分布：dataallocation_1,classes_pc_5
write_data()     # 写入空行
NonIID_FL_5 = NonIIDlearning_1.FLlearningNonIID_1()
NonIID_FL_5.dataallocation = dataallocation_1
NonIID_FL_5.num_rounds = num_rounds
NonIID_FL_5.classes_pc = classes_pc_5
y_acctest_5, y_losstest_5 = NonIID_FL_5.main()
print("Non_IID训练：{0}数据分配的情况下,训练完成".format(dataallocation_1))
NonIID_FL_5.csvnum = 5
write_data(y_acctest_5)
del NonIID_FL_5

# 非独立同分布：dataallocation_1,classes_pc_7
write_data()      # 写入空行
NonIID_FL_7 = NonIIDlearning_1.FLlearningNonIID_1()
NonIID_FL_7.dataallocation = dataallocation_1
NonIID_FL_7.num_rounds = num_rounds
NonIID_FL_7.classes_pc = classes_pc_7
y_acctest_7, y_losstest_7 = NonIID_FL_7.main()
print("Non_IID训练：{0}数据分配的情况下,训练完成".format(dataallocation_1))
NonIID_FL_7.csvnum = 7
write_data(y_acctest_7)
del NonIID_FL_7

# 非独立同分布：dataallocation_1,classes_pc_9
write_data()
NonIID_FL_9 = NonIIDlearning_1.FLlearningNonIID_1()
NonIID_FL_9.dataallocation = dataallocation_1
NonIID_FL_9.num_rounds = num_rounds
NonIID_FL_9.classes_pc = classes_pc_9
y_acctest_9, y_losstest_9 = NonIID_FL_9.main()
print("Non_IID训练：{0}数据分配的情况下,训练完成".format(dataallocation_1))
NonIID_FL_9.csvnum = 9
write_data(y_acctest_9)
del NonIID_FL_9

# 独立同分布：dataallocation_1
IID_FL = IIDlearning.FLLearning()
IID_FL.dataallocation = dataallocation_1
IID_FL.num_rounds = num_rounds
y_acctest_iid, y_losstest_iid = IID_FL.main()
print("IID训练：{0}数据分配的情况下，训练完成".format(dataallocation_1))
IID_FL.csvnum = 10
write_data(y_acctest_iid)
del IID_FL

plt.figure() # 新建一个窗口
# acc test
plt.title('Classes in each client')
plt.plot(x, y_acctest_1, color='blue', label="class=1")
plt.plot(x, y_acctest_2, color='green', label="classes=2")
plt.plot(x, y_acctest_3,  color='red', label="classes=3")
plt.plot(x, y_acctest_5, color='cyan', label="classes=5")
plt.plot(x, y_acctest_7, color='magenta', label="classes=7")
plt.plot(x, y_acctest_9, color='yellow', label="classes=9")
plt.plot(x, y_acctest_iid, color='black', label="IID")
plt.xlabel('Communication Rounds')
plt.ylabel('Test Accuracy')
plt.legend()    # 画出线lable
plt.gcf()       # 创建网格
plt.savefig('./PaperFigureSave/test_classes_pc.jpg')    # 保存图片
plt.show()