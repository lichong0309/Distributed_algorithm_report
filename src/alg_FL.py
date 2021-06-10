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

TrustMatrix = [[0,1,1,1,1,1,1,1,1,1],
               [1,0,1,1,1,1,1,1,1,1],
               [1,1,0,1,1,1,1,1,1,1],
               [1,1,1,0,1,1,1,1,1,1],
               [1,1,1,1,0,1,1,1,1,1],
               [1,1,1,1,1,0,1,1,1,1],
               [1,1,1,1,1,1,0,1,1,1],
               [1,1,1,1,1,1,1,0,1,1],
               [1,1,1,1,1,1,1,1,0,1],
               [1,1,1,1,1,1,1,1,1,0]]

# 类的继承，子类alg_FL继承父类NonIIDlearning_1.FLlearningNonIID_1
class alg_FL(NonIIDlearning_1.FLlearningNonIID_1):
    def __init__(self) -> None:
        NonIIDlearning_1.FLlearningNonIID_1.__init__(self)   # 继承父类的初始化函数
        # super(alg_FL, self).__init__(dataallocation)
        self.TrustMatrix = TrustMatrix     # 信任矩阵
        self.dataallocation = config.get("DataAllocation_13")

    def clients_rand(self, train_len, nclients):
        client_dist = []     
        for i in range(len(self.dataallocation) -1):
            dtmp = int(train_len * self.dataallocation[i])
            client_dist.append(dtmp)
        client_dist_array = np.array(client_dist)       # 将client_dist转化为np.array类型,使用下面的sum函数计算
        num = train_len - client_dist_array.sum()       # 计算剩余的数据量
        client_dist.append(num)           # 将剩余的数据量加入list中
        to_ret = client_dist
        return to_ret
        

    def get_client_matrix_list(self):
        # 信任矩阵转换的client之间的相关关系list
        # list_temp = []              # 初始化为空list
        client_matrix_list = []    # 存放每个client相邻的client的编号
        for i in range(self.num_clients):
            client_matrix_list.append([])   # eg. [[2,3],[1,2],...,[4,6]]
        for i in range(self.num_clients):
            for j in range(len(self.TrustMatrix[i])):
                if i != j:
                    if self.TrustMatrix[i][j] == 1:
                        client_matrix_list[i].append(j)
                    else:
                        pass
                else:
                    pass
        # print(self.TrustMatrix, self.TrustMatrix[0])
        # print(client_matrix_list[0])
        print("client之间的相关关系client_matrix_list:",client_matrix_list)
        return client_matrix_list

    # 获得最小值
    def client_min(self, data_per_client_temp, operated_list):
        client_min_data_num = 0   # 初始化为0 
        client_min_data_index = 0 # 初始化为编号为0的client
        for i in range(len(data_per_client_temp)):
            if i in operated_list:
                pass
            else:
                client_min_data_num = data_per_client_temp[i]
                client_min_data_index = i
        for i in range(len(data_per_client_temp)):
            if data_per_client_temp[i] < client_min_data_num:
                if i in operated_list:
                    pass
                else:
                    client_min_data_num = data_per_client_temp[i]
                    client_min_data_index = i
            else:
                pass
    
        return client_min_data_num, client_min_data_index 


    # 获得最大值
    def client_max(self, client_matrix_data_list, operated_list, client_signal_matrix_list):
        client_max_data_num = 0  # 初始化为0 
        client_max_data_index = 0 # 初始化为编号为0的client
        for i in range(len(client_matrix_data_list)):
            if client_signal_matrix_list[i] in operated_list:
                pass
            else:
                # 初始化client_max_data_num为第一个不在operated_list中对应的数据和索引
                client_max_data_num = client_matrix_data_list[i]
                client_max_data_index = client_signal_matrix_list[i]
                break  

        for i in range(len(client_matrix_data_list)):
            if client_matrix_data_list[i] > client_max_data_num:
                if client_signal_matrix_list[i] in operated_list:          # 如果在operated_list中，则不做任何操作
                    pass
                else:
                    client_max_data_num = client_matrix_data_list[i]
                    client_max_data_index = client_signal_matrix_list[i]
            else:
                pass

        return client_max_data_num, client_max_data_index


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
        data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]
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
        
        # split data among clients
        clients_split = []
        client_split_original = []          # original:没有进行传输之前的数据分配
        c = 0
        for i in range(n_clients):
            # c = 0
            client_idcs = []

            budget = data_per_client_temp[i]                 # 每个client的数据量

            c = np.random.randint(n_labels)             # 随机获得class
            while budget > 0:
                take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)  # 得到参数最小值
                
                client_idcs += data_idcs[c][:take]
                data_idcs[c] = data_idcs[c][take:]
                
                budget -= take

                c = (c + 1) % n_labels                  # 获取新的class
                # c = c + 1

            client_split_original.append(client_idcs)

        client_split_after = self.communicate_split_data(client_split_original, data_per_client_temp)
        for i in range(self.num_clients):
            clients_split += [(data[client_split_after[i]], labels[client_split_after[i]])]
            
             
            # clients_split += [(data[client_idcs], labels[client_idcs])]

        print("data_per_client:1:",data_per_client)  
        print("data_per_client_temp:1:",data_per_client_temp)   
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

    def del_list(self, client_matrix_list, operated_list):
        # 消除和其他任何一个client都没有联系的节点client
        for i in range(len(client_matrix_list)):
            if len(client_matrix_list[i]) == 0:
                operated_list.append(i)
            else:
                flag = 0 
                for j in range(len(client_matrix_list[i])):
                    if client_matrix_list[i][j] in operated_list:
                        flag = 0 
                    else:
                        flag = 1
                if flag == 0:
                    operated_list.append(i)
                else:
                    pass
        operated_list = list(set(operated_list))    # 去除重复元素
        return operated_list                    

    
    def communicate_split_data(self, client_split_original, data_per_client_temp):

        client_matrix_list = self.get_client_matrix_list()    # 得到信任矩阵转换的client之间的信任关系
        client_split_original_temp = copy.deepcopy(client_split_original)       # 深拷贝
        operated_list = []   # 存放已经操作的元素
        operated_list = self.del_list(client_matrix_list, operated_list)
        while len(operated_list) != self.num_clients:
            operated_list = self.del_list(client_matrix_list, operated_list)

            # 数据按照信任矩阵进行传输:
            # 获得最小值
            client_min_data_num, client_min_data_index = self.client_min(data_per_client_temp, operated_list)
            # 检查是否有其他的最小值：
            for i in range(len(data_per_client_temp)):
                if data_per_client_temp[i] == client_min_data_num:
                    if i not in operated_list:
                        # 如果新的最小值的关联值小于当前最小值的关联值，更新最小值的索引
                        if len(client_matrix_list[i]) < len(client_matrix_list[client_min_data_index]):
                            client_min_data_index = i
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            
            # 获得与节点client相连的节点的数据量最大值
            client_matrix_data_list = []    # 存放与节点client相连的节点的数据量list
            for i in client_matrix_list[client_min_data_index]:
                client_matrix_data_list.append(data_per_client_temp[i])
            client_max_data_num, client_max_data_index = self.client_max(client_matrix_data_list, operated_list, client_matrix_list[client_min_data_index])
            # 检查是否有其他的最大值：
            for i in client_matrix_list[client_min_data_index]:
                if data_per_client_temp[i] == client_max_data_num:
                    if i not in operated_list:
                        # 如果新的最大值的关联值小于当前最大值的关联值，更新最大值的索引
                        if len(client_matrix_list[i]) < len(client_matrix_list[client_max_data_index]):
                            client_max_data_index = i
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            
            # 传输
            # 传输的数据量
            communicate_num = int((data_per_client_temp[client_min_data_index] + data_per_client_temp[client_max_data_index]) / 2) - data_per_client_temp[client_min_data_index]
            data_per_client_temp[client_max_data_index] -= communicate_num
            data_per_client_temp[client_min_data_index] += communicate_num
            # 随机选择传输的数据
            random.seed(10)
            communicate_list = random.sample(client_split_original_temp[client_max_data_index],communicate_num)
            # 对于接收数据的client
            client_split_original_temp[client_min_data_index] += communicate_list
            # 对于传输数据的client
            # temp_list = copy.deepcopy(client_split_original_temp)   # 深拷贝
            # for list_ in communicate_list:
            #     for list_2 in range(len(client_split_original_temp[client_max_data_index])):
            #         if list_ == client_split_original_temp[client_max_data_index][list_2]:
            #             del client_split_original_temp[client_max_data_index][list_2]
            #         else:
            #             pass
            for list_ in communicate_list:
                del_index = client_split_original_temp[client_max_data_index].index(list_)    # 寻找索引
                del client_split_original_temp[client_max_data_index][del_index]
            operated_list.append(client_min_data_index)   # 不参与下次的迭代
            operated_list = self.del_list(client_matrix_list, operated_list)
        client_split_after = client_split_original_temp
        return client_split_after

    def main(self):
        ############################################
        #### Initializing models and optimizer  ####
        ############################################

        #### global model ##########
        global_model =  VGG('VGG19').cuda()

        ############# client models ###############################
        client_models = [ VGG('VGG19').cuda() for _ in range(self.num_selected)]
        for model in client_models:
            model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global modle 

        ###### optimizers ################
        opt = [optim.SGD(model.parameters(), lr = 0.01) for model in client_models]

        ####### baseline data ############
        loader_fixed = self.baseline_data(self.baseline_num)


        ###### Loading the data using the above function ######
        train_loader, test_loader = self.get_data_loaders(classes_pc=self.classes_pc, nclients= self.num_clients,
                                                            batch_size=self.batch_size,verbose=True)

        losses_train = []
        losses_test = []
        acc_test = []
        losses_retrain=[]

        # Runnining FL
        for r in range(self.num_rounds):    #Communication round
            # select random clients
            client_idx = np.random.permutation(self.num_clients)[:self.num_selected]
            client_lens = [len(train_loader[idx]) for idx in client_idx]

            # client update
            loss = 0
            for i in tqdm(range(self.num_selected)):
                self.client_syn(client_models[i], global_model)
                loss += self.client_update(client_models[i], opt[i], train_loader[client_idx[i]], self.epochs,model)
            losses_train.append(loss)

            # server aggregate
            #### retraining on the global server
            loss_retrain = 0
            for i in tqdm(range(self.num_selected)):
                loss_retrain+= self.client_update(client_models[i], opt[i], loader_fixed, self.retrain_epochs,model)
            losses_retrain.append(loss_retrain)
            
            ### Aggregating the models
            self.server_aggregate(global_model, client_models,client_lens)
            test_loss, acc = self.test(global_model, test_loader,model)
            losses_test.append(test_loss)
            acc_test.append(acc)
            print('%d-th round' % r)
            print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss_retrain / self.num_selected, test_loss, acc))
        return acc_test, losses_test

    
    def __del__(self):
        print("alg_FL对象被收回.....")


if __name__ == "__main__":
    # 信任矩阵
    TrustMatrix = [[0,1,1,1,1,1,1,1,1,1],
                   [1,0,1,1,1,1,1,1,1,1],
                   [1,1,0,1,1,1,1,1,1,1],
                   [1,1,1,0,1,1,1,1,1,1],
                   [1,1,1,1,0,1,1,1,1,1],
                   [1,1,1,1,1,0,1,1,1,1],
                   [1,1,1,1,1,1,0,1,1,1],
                   [1,1,1,1,1,1,1,0,1,1],
                   [1,1,1,1,1,1,1,1,0,1],
                   [1,1,1,1,1,1,1,1,1,0]]
    dataallocation_2 = config.get("DataAllocation_13")      # [0.075, 0.125, 0.025, 0.075, 0.2, 0.16, 0.05, 0.14, 0.1, 0.05]
    class_pc = 4
    alg_FL = alg_FL()
    print("class_pc",alg_FL.classes_pc)
    alg_FL.TrustMatrix = TrustMatrix
    alg_FL.dataallocation = dataallocation_2
    alg_FL.classes_pc = class_pc
    print("class_pc_1",alg_FL.classes_pc)
    acc_test , loss_test = alg_FL.main()



    