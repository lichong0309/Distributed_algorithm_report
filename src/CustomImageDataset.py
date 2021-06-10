# 将分割转换为数据加载器，图像增强，用于将其作为输入提供给模型进行训练。
import os
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

class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''
    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms 

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)
        return (img, label)     # 返回元组(tuple),元组与list相似，但是元组不能修改，元组使用小括号。
     
    def __len__(self):
        return self.inputs.shape[0]