from IIDlearning import dataallocation
import os
import copy
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms 
from torchvision.transforms import Compose 
torch.backends.cudnn.benchmark=True
from configuration import config
from CustomImageDataset import CustomImageDataset     # 图像增强模块
from VGGModule import VGG                   # 神经网络模块 
from dataallocationrandom import dataallocationrandom
import NonIIDlearning_1

# 类的继承，子类alg_FL继承父类NonIIDlearning_1.FLlearningNonIID_1
class alg_FL(NonIIDlearning_1.FLlearningNonIID_1):
    def __init__(self,dataallocation, TrustMatrix) -> None:
        NonIIDlearning_1.FLlearningNonIID_1.__init__(self, dataallocation)   # 继承父类的初始化函数
        # super(alg_FL, self).__init__(dataallocation)
        self.TrustMatrix = TrustMatrix     # 信任矩阵

    def clients_rand(self, train_len, nclients):
        super().clients_rand(train_len, nclients)     # 继承父类的方法clients_rand

    # 重写父类函数
    def split_image_data(self, data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
        # 判断self.dataallocation中的元素是否都为平均值，即每个client中的数据量为平均分配
        # flag = 1: 平均分配，flag = 0 :非平均分配
        print("self.dataallocation:",self.dataallocation)
        print("ave:",1/self.num_clients)
        
        flag = 1      # 初始化flag = 1
        for i in self.dataallocation:
            if i != (1 / self.num_clients):
                flag = 0 
            else:
                pass
            
        if flag == 1:
            print("flag = 1,数据平均分配")
        else:
            print("flag = 0，数据非平均分配")
        
        #### constants #### 
        n_data = data.shape[0]                  # 总的数据量
        n_labels = np.max(labels) + 1           # 求序列的最值，即为n_labels的数量

        ### client distribution ####
        data_per_client = self.clients_rand(len(data), n_clients) 
        print("data_per_client,环境中主机分配到的数据量大小为:",data_per_client)         
        data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]
        print("data_per_client_per_class，环境中每个client中的每个类最多能分配到的数据大小:",data_per_client_per_class)

        data_per_client_temp = copy.deepcopy(data_per_client)       # 深拷贝

        # data temp
        if flag == 1:
            data_per_client_temp[(len(self.dataallocation) - 1)] -= 1
        else:
            pass
        print("data_per_client_temp:",data_per_client_temp)

        # sort for labels
        data_idcs = [[] for i in range(n_labels)]
        for j, label in enumerate(labels):
            data_idcs[label] += [j]
        if shuffle:
            for idcs in data_idcs:
                np.random.shuffle(idcs)

        # 参数
        label_list_temp = []              # 辅助list e.g.[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],每个元素对应于数据集的class，0表示client中没有此class，1表示client中存在此class
        for i in range(n_labels):
            label_list_temp.append(0)               # 初始化为0
        print("辅助函数label_list_temp:",label_list_temp)

        # original：每个client中包含的class，如果为0，表示client中存在该class，如果为1，表示client中不存在该class
        client_class_kind_original_list = []      # e.g. [[0,1,0],[0,0,1],...,[1,0,0]]
        for i in range(self.num_clients):
            client_class_kind_original_list.append(label_list_temp)         # 初始化为0
        print("client_class_kind__original_list:",client_class_kind_original_list)

        # clients中每个class包含的数量：
        client_class_num_original_list = []  # e.g. [[0,10,0], [0,0,20],... , [30,0,0]]
        for i in range(self.num_clients):
            client_class_kind_original_list.append(label_list_temp)    # 初始化为0
        print("client_class_kind_original_list:",client_class_num_original_list)

        # client中每个class的data, 每个元素是字典
        client_class_data_original_list = []   # e.g [{"1":"1"}, {"2":"2"},... ,{"0":"1"}]
        dic_temp_list = {}      # 空字典
        for i in range(self.num_clients):
            client_class_data_original_list.append(dic_temp_list)
        print("client_class_data_original_list:",client_class_data_original_list)


        # split data among clients
        clients_split = []
        c = 0
        for i in range(n_clients):
            # c = 0
            client_idcs = []

            budget = data_per_client_temp[i]                 # 每个client的数据量

            c = np.random.randint(n_labels)             # 随机获得class

            client_class_kind_original_list[i][c] = 1      # 1表示client中存在该class
            
            while budget > 0:
                take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)  # 得到参数最小值
                
                # 更新client中该class的数量
                client_class_num_original_list[i][c] = client_class_num_original_list[i][c] + take

                client_idcs += data_idcs[c][:take]
                data_idcs[c] = data_idcs[c][take:]
                
                # 更新client中该class的数据data
                # 检测字典中是否存有数据，判断字典中是否存在key
                keys_in = client_class_data_original_list[i].has_key("c")
                if keys_in == True:
                    client_class_data_original_list[i]['c'] += client_idcs   # 在原先的上面增加
                else:
                    client_class_data_original_list[i]['c'] = client_idcs   # 直接添加字典元素
            
                budget -= take
                
                c = (c + 1) % n_labels                  # 获取新的class
                # c = c + 1

            clients_split += [(data[client_idcs], labels[client_idcs])]


        # 内嵌函数，外部无法访问，打印信息
        def print_split(clients_split):
            print("Data split:")
            for i, client in enumerate(clients_split):
                split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
                print(" - Client {}: {}".format(i,split))
            print()
        
            if verbose:
                print_split(clients_split)

        clients_split = np.array(clients_split,dtype=object)

        return clients_split



    def alg_split_data(self,client_class_kind_original_list, client_class_num_original_list, client_class_data_original_list):
        list_temp = []   # 空list

        client_matrix_list = []    # 存放每个client相邻的client的编号
        for i in range(self.num_clients):
            client_matrix_list.append(list_temp)   # eg. [[2,3],[1,2],...,[4,6]]
        for i in range(self.num_clients):
            for j in self.TrustMatrix[i]:
                if self.TrustMatrix[i][j] == 1:
                    client_matrix_list[i].append(j)
                else:
                    pass
        print("client_matrix_list:",client_matrix_list)

        # 每个client中class的编号
        client_class_kinds = []
        for i in range(self.num_clients):
            client_class_kinds.append(list_temp)
        for i in range(self.num_clients):           # client_class_kinds中
            for j in range(len(client_class_kind_original_list[i])):
                if client_class_kind_original_list[i][j] == 1:
                    client_class_kinds[i].append(j)   # 添加到client_class_kinds中
                else:
                    pass

        # 每个client中每个class的请求
        list_temp_1 = []
        client_class_request = [] 
        for i in range(len(client_class_kind_original_list[0])):
            list_temp_1.append(0)       # 初始化为0 
        for i in range(self.num_clients):
            client_class_request.append(list_temp_1)      # eg. [[0,0,0], [0,0,0],...,[0,0,0]]
        

        


        









        return client_class_num_after_list, client_class_num_after_list, client_class_data_after_list

    def main(self):
        super(alg_FL, self).main()    # 调用父类的方法main()

if __name__ == "__main__":
    alg_FL = alg_FL(dataallocation, TrustMatrix)
    