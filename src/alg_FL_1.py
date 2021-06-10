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

class alg_FL_1(NonIIDlearning_1.FLlearningNonIID_1):
    def __init__(self):
        super().__init__()
        self.TrustMatrix = TrustMatrix
        self.dataallocation = config.get("DataAllocation_13")

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

    # 重写父类
    def split_image_data(self, data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
        pass

if __name__ = "__main__":
    pass
