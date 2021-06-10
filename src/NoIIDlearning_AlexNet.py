# 聚合之前在服务器中进行在训练(baseline number = 100)
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
from dataallocationrandom import dataallocationrandom
from AlexNetModule import AlexNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FLlearningNonIID_1(object):
    def __init__(self):
        self.dataallocation = config.get("DataAllocation_14")     # 每个主机的分配的数据list
        self.classes_pc = 1                             # 每个主机分配的class的大小
        self.num_clients = len(self.dataallocation)          # 环境中主机的个数
        self.num_selected = self.num_clients            # 进行训练的主机的个数
        self.num_rounds = 20                           # 主机和服务器之间进行通信的轮数    
        self.epochs = 5                                 # client本地训练的轮数
        self.batch_size = 32                            # 将数据批量加载到数据加载器中
        self.baseline_num = 100                         # 服务器重新训练的基准线
        self.retrain_epochs = 20                        # 服务器重新训练的轮数
        self.del_()                                     # 消除数据为0的节点client

    def del_(self):
        dataallocation_i_temp = [] # 存放元素不为0的元素
        for i in range(len(self.dataallocation)):
            if self.dataallocation[i] != 0:
                dataallocation_i_temp.append(self.dataallocation[i])
            else:
                pass
        self.dataallocation = dataallocation_i_temp 
        # 更新self.num_clients和num_selected
        self.num_clients = len(self.dataallocation)
        if self.num_selected > self.num_clients:
            # self.num_selected = random.randint(1,self.num_clients)
            self.num_selected = self.num_clients 
        else:
            pass


    #### get cifar dataset in x and y form
    # 得到cifar数据集的数据和标签
    def get_cifar10(self):
        '''Return CIFAR10 train/test data and labels as numpy arrays'''
        data_train = torchvision.datasets.CIFAR10('./data', train=True, download=True)
        data_test = torchvision.datasets.CIFAR10('./data', train=False, download=True) 
        
        x_train, y_train = data_train.data.transpose((0,3,1,2)), np.array(data_train.targets)
        x_test, y_test = data_test.data.transpose((0,3,1,2)), np.array(data_test.targets)

        return x_train, y_train, x_test, y_test
    

    def print_image_data_stats(self,data_train, labels_train, data_test, labels_test):
        print("\nData: ")
        print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
            data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
            np.min(labels_train), np.max(labels_train)))
        print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
            data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
            np.min(labels_test), np.max(labels_test)))
    
    
    def clients_rand(self,train_len, nclients):
        '''
        train_len: size of the train data
        nclients: number of clients
        
        Returns: to_ret
        
        This function creates a random distribution 
        for the clients, i.e. number of images each client 
        possess.
        # '''
        # # 随机给client数据比例random
        # client_tmp=[]
        # sum_=0
        # #### creating random values for each client ####
        # for i in range(nclients-1):                     # 把剩余的数据量给最后一个client
        #     tmp=random.randint(10,100)
        #     sum_+=tmp
        #     client_tmp.append(tmp)

        # client_tmp= np.array(client_tmp)
        # #### using those random values as weights ####
        # clients_dist= ((client_tmp/sum_)*train_len).astype(int)
        # num  = train_len - clients_dist.sum()             # 计算剩余的数据量
        # to_ret = list(clients_dist)                 
        # to_ret.append(num)                          # 将剩余的数据量加入list中  
        # return to_ret
         
        # client的数据比例给定dataallocation  
        print("test_1")
        client_dist = []     
        for i in range(len(self.dataallocation) -1):
            dtmp = int(train_len * self.dataallocation[i])
            client_dist.append(dtmp)
        client_dist_array = np.array(client_dist)       # 将client_dist转化为np.array类型,使用下面的sum函数计算
        num = train_len - client_dist_array.sum()       # 计算剩余的数据量
        client_dist.append(num)           # 将剩余的数据量加入list中
        to_ret = client_dist
        print("test_2")
        return to_ret
        

    def split_image_data(self,data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
        '''
        Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
        Input:
            data : [n_data x shape]
            labels : [n_data (x 1)] from 0 to n_labels
            n_clients : number of clients
            classes_per_client : number of classes per client
            shuffle : True/False => True for shuffling the dataset, False otherwise
            verbose : True/False => True for printing some info, False otherwise
        Output:
            clients_split : client data into desired format
        '''

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

        data_per_client_temp = copy.deepcopy(data_per_client)
        
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
        
            clients_split += [(data[client_idcs], labels[client_idcs])]
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


    def shuffle_list(self,data):
        '''
        This function returns the shuffled data
        '''
        # print("len(data)",len(data))
        for i in range(len(data)):
            tmp_len = len(data[i][0])
            index = [i for i in range(tmp_len)]
            random.shuffle(index)
            data[i][0],data[i][1] = self.shuffle_list_data(data[i][0],data[i][1])
        return data

    def shuffle_list_data(self,x, y):
        '''
        This function is a helper function, shuffles an
        array while maintaining the mapping between x and y
        '''
        inds = list(range(len(x)))
        random.shuffle(inds)
        return x[inds],y[inds]
            
    def get_default_data_transforms(self,train=True, verbose=True):
        transforms_train = {
        'cifar10' : transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
        }
        transforms_eval = {    
        'cifar10' : transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        }
        if verbose:
            print("\nData preprocessing: ")
            for transformation in transforms_train['cifar10'].transforms:
                print(' -', transformation)
            print()

        return (transforms_train['cifar10'], transforms_eval['cifar10'])

    def get_data_loaders(self,nclients,batch_size,classes_pc=10 ,verbose=True ):
        
        x_train, y_train, x_test, y_test = self.get_cifar10()

        if verbose:
            self.print_image_data_stats(x_train, y_train, x_test, y_test)

        transforms_train, transforms_eval = self.get_default_data_transforms(verbose=False)
        
        split = self.split_image_data(x_train, y_train, n_clients=nclients, 
                classes_per_client=self.classes_pc, verbose=verbose)
        # print("n_client test:",nclients)
        split_tmp = self.shuffle_list(split)
        
        client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), 
                                                                        batch_size=batch_size, shuffle=True) for x, y in split_tmp]

        test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False) 

        return client_loaders, test_loader

    def baseline_data(self,num):
        '''
        Returns baseline data loader to be used on retraining on global server
        Input:
                num : size of baseline data
        Output:
                loader: baseline data loader
        '''
        xtrain, ytrain, xtmp, ytmp = self.get_cifar10()
        x , y = self.shuffle_list_data(xtrain, ytrain)

        x, y = x[:num], y[:num]
        transform, _ = self.get_default_data_transforms(train=True, verbose=False)
        loader = torch.utils.data.DataLoader(CustomImageDataset(x, y, transform), batch_size=16, shuffle=True)
        return loader

    def client_update(self, client_model, optimizer, train_loader, epoch, model):
        """
        This function updates/trains client model on client data
        """
        model.train()
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                # data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = client_model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
        return loss.item()


    def client_syn(self, client_model, global_model):
        '''
        This function synchronizes the client model with global model
        '''
        client_model.load_state_dict(global_model.state_dict())


    def server_aggregate(self, global_model, client_models, client_lens):
        """
        This function has aggregation method 'wmean'
        wmean takes the weighted mean of the weights of models
        """
        total = sum(client_lens)
        n = len(client_models)
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float()*(n*client_lens[i]/total) for i in range(len(client_models))], 0).mean(0)
        global_model.load_state_dict(global_dict)
        for model in client_models:
            model.load_state_dict(global_model.state_dict())


    def test(self, global_model, test_loader, model):
        """
        This function test the global model on test 
        data and returns test loss and test accuracy 
        """
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                # data, target = data.to(device), target.to(device)
                output = global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset)

        return test_loss, acc

    def main(self):
        print("test")
        ############################################
        #### Initializing models and optimizer  ####
        ############################################

        #### global model ##########
        global_model =  AlexNet().cuda()

        ############# client models ###############################
        client_models = [ AlexNet().cuda() for _ in range(self.num_selected)]
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

    # 类的销毁
    def __del__(self):
        print("FLLearningNon-IID的对象被收回----\n")

if __name__ == "__main__":
    dataallocation = config.get("DataAllocation_14")
    # dataallocation = dataallocationrandom(10)
    FL = FLlearningNonIID_1()
    FL.num_rounds = 2
    acc_test, losses_test = FL.main()
