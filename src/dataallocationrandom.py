import random
import numpy as np
from configuration import config
def dataallocationrandom(nclients):
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

    client_tmp = []      # 存放生成的随机数
    sum_ = 0
    # 产生nclients个随机数
    for i in range(nclients):
        tmp = random.randint(10,100)    # 产生10到100之间的随机数
        sum_ = sum_ + tmp
        client_tmp.append(tmp)              # 把随机数加入list中

    # 归一化处理
    for i in range(len(client_tmp)):
        client_tmp[i] = client_tmp[i] / sum_
    dataallocationrandom = client_tmp
    print("分配的数据为：{}".format(dataallocationrandom))
    
    return dataallocationrandom