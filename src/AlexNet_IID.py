# 独立同分布的联邦学习
# 导入包
import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from configuration import config
from VGGModule import VGG
from dataallocationrandom import dataallocationrandom
from AlexNetModule import AlexNet
torch.backends.cudnn.benchmark=True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AlexNetFLLearning(object): 
    def __init__(self):
        # 参数设置
        self.dataallocation = config.get("DataAllocation_14") # 获得各个主机分配的数据比例
        self.num_clients = len(self.dataallocation)           # 环境中主机的数量
        self.num_selected = self.num_clients                  # 环境中有数据的主机的数量
        self.num_rounds = 20                                  # 选定的主机之间的通讯总次数
        self.epochs = 5                                       # 本地训练次数
        self.batch_size = 32                                  # 将数据批量加载到数据加载器中
        self.lr = 0.01                                        # 学习率
        self.DataUse = 1.0                                    # 使用总的CIFAR数据量


    # 下载和加载数据集
    def datadownlowd(self):
        # 图像增强
        # transforms包含了一些常用的图像变换，这些变换能够用Compose串联组合起来
        transform_train = transforms.Compose([            
            transforms.RandomCrop(32, padding=4),   # 随机裁剪，依据给定的size=32随机裁剪，
            transforms.RandomHorizontalFlip(),       # 依据概率p对PIL图片进行水平翻转,默认p=0.5
            transforms.ToTensor(),                  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        ###############################################################
        # 加载数据集
        self.traindata = datasets.CIFAR10('./data', train=True, download=True,
                        transform= transform_train)
        print("data loda finish...........")


    # 在客户机之间创建和分配数据分布
    # 将数据分配给各个client
    def ClientDataAllocation(self):
        # 检查self.dataallocation 中是否存在数据等于0
        # 如果数据等于0，删除该数据，并且更新self.num_client和self.num_selected
        dataallocation_i_temp = []          # 定义临时list，存放不为0的元素
        for i in range(len(self.dataallocation)):
            if self.dataallocation[i] != 0:
                dataallocation_i_temp.append(self.dataallocation[i])     # 不为0的元素加入到临时list中
            else:
                pass
        self.dataallocation = dataallocation_i_temp         # self.dataallocation等于临时list
        
        # 更新self.num_clients和num_selected
        self.num_clients = len(self.dataallocation)     
        if self.num_selected > self.num_clients:
            # self.num_selected = random.randint(1,self.num_selected)
            self.num_selected = self.num_clients
        else:
            pass

        print("数据开始分配给各个client...........")
        datatemp = []
        c = 0
        for i in range(self.num_clients):
            temp = int(self.traindata.data.shape[0] * self.dataallocation[i] * self.DataUse)   # 分数据
            datatemp.append(temp)
            c = c + temp
        ctemp = self.traindata.data.shape[0] - c      # 剩余数据的数据大小
        datatemp.append(ctemp)       # 将剩余数据存放在list最后一位
        traindata_split = torch.utils.data.random_split(self.traindata,datatemp)    # 分割数据集
        del traindata_split[self.num_clients]       # 从list中del剩余数据ctemp
        print("数据分配完成............")
        return traindata_split
    
    def data_loader(self,traindata_split):
        # 为训练模型创建Pytorch加载程序
        self.train_loader = [torch.utils.data.DataLoader(x, batch_size=self.batch_size, \
            shuffle=False) for x in traindata_split]

        # 标准化测试图像
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # 加载测试图，然后将它们转换为test_loader
        self.test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                ), batch_size=self.batch_size, shuffle=True)

    ###########
    # 联合训练的辅助函数
    # 在本机client中训练本地数据
    def client_update(self, client_model, optimizer, train_loader, epoch, model):
        """
        This function updates/trains client model on client data
        """
        criterion = nn.CrossEntropyLoss()
        model.train()
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()                               
                # data, target = data.to(device), target.to(device)
                optimizer.zero_grad()                                      # 归零梯度
                output = client_model(data)                        # 得到前向传播的结果
                loss = criterion(output, target)                   # 得到损失函数
                loss.backward()                             # 反向传播
                optimizer.step()                               # 优化
        return loss.item()

    # server_aggregate函数聚合从每个客户端接收的模型权重，并使用更新后的权重更新全局模型。
    def server_aggregate(self, global_model, client_models,client_dataallocation):
        global_dict = global_model.state_dict()
        if self.dataallocation[0] == 1/self.num_clients or self.dataallocation[0] == 1.0:
            for k in global_dict.keys():
                global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() \
                    for i in range(len(client_models))], 0).mean(0)
        else:
            for k in global_dict.keys():
                a = client_models[0].state_dict()[k].float() * client_dataallocation[0] # 初始化a为client0的参数
                for i in range(len(client_models)):              
                    if i == 0:
                        pass
                    else:
                        b = client_models[i].state_dict()[k].float() * client_dataallocation[i]
                        a = torch.add(a,b)        # Torch类型数据 a, b 相加
                global_dict[k] =  a   
        global_model.load_state_dict(global_dict)
        for model in client_models:
            model.load_state_dict(global_model.state_dict())


    # 测试函数
    # 返回test_loss和acc
    def test(self, global_model, test_loader, model):
        """
        This function test the global model on test 
        data and returns test loss and test accuracy 
        """
        criterion = nn.CrossEntropyLoss()
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():                               # 训练集中不需要反向传播
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()                               # 将输入和目标在每一步都送入GPU
                # data, target = data.to(device), target.to(device)
                output = global_model(data)                                                     
                test_loss += criterion(output, target).item()            # sum up batch loss
                # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # correct += pred.eq(target.view_as(pred)).sum().item()
                _, predicted = torch.max(output.data, 1)                            # 返回每一行中最大值的那个元素，且返回其索引值
                correct +=  (predicted == target).sum()

        # test_loss /= len(test_loader.dataset)
        acc =  correct / len(test_loader.dataset)

        return test_loss, acc



    def main(self):
        self.datadownlowd()   # 下载和加载数据
        traindata_split = self.ClientDataAllocation()     # 给每个client分配数据
        self.data_loader(traindata_split)       # 加载训练集和测试集
        
        # 初始化模型和优化器
        # 全局模型
        global_model =  AlexNet().cuda()
        # client模型
        client_models = [ AlexNet().cuda() for _ in range(self.num_selected)]
        for model in client_models:
            model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global model
        # 优化器
        opt = [optim.SGD(model.parameters(), lr=self.lr) for model in client_models]

        # 学习信息的list
        losses_train = []
        losses_test = []
        acc_train = []
        acc_test = []

        # 训练fl
        for r in range(self.num_rounds):
            # select random clients,随机选择环境中的主机
            client_idx = np.random.permutation(self.num_clients)[:self.num_selected]    # 生成随机序列
            print("client_idx，被选中的主机的编号为：",client_idx)
            # 获取和更新selected的client对应的数据分配
            client_dataallocation = [self.dataallocation[idx] for idx in client_idx]     # 获取selected的client对应的数据分配
            sum_ = sum(client_dataallocation)           # b被选中的selected_client的数据比例总和
            # 更新client_dataallocation，即被选中的client的数据分配，归一化
            for cd in range(len(client_dataallocation)):
                client_dataallocation[cd] =  client_dataallocation[cd] / sum_
            print("self.dataallocation，更新前环境中的主机的数据占比:",self.dataallocation)
            print("client_dataallocation，更新后的被选择的主机的数据占比：",client_dataallocation)

            # client update
            losstemp = 0
            for i in tqdm(range(self.num_selected)):
                losstemp += self.client_update(client_models[i], opt[i], self.train_loader[client_idx[i]],self.epochs,model)
            losses_train.append(losstemp)
            
            # server aggregate
            self.server_aggregate(global_model, client_models,client_dataallocation)
            
            test_loss, acc = self.test(global_model, self.test_loader,model)
            losses_test.append(test_loss)
            acc_test.append(acc)
            print('%d-th round' % r)
            print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (losstemp / self.num_selected, test_loss, acc))
        
        # del global_model    # 收回实例化的VGG类
        return acc_test, losses_test

    # 类的销毁
    def __del__(self):
        print("FLLearning的对象被收回----\n")

if __name__ == "__main__":
    # clientnum = config.get("ClientNum_0")
    dataallocation = config.get("DataAllocation_14")
    # # dataallocation  = dataallocationrandom(20)
    # epochs = config.get("epochs_4")
    # datause = 1.0
    FL = AlexNetFLLearning()
    FL.dataallocation = dataallocation
    FL.num_rounds = 100
    acc_test , losses_test = FL.main()



