# 随机生成数据分配
import random
import numpy as np
from configuration import config
from dataallocationrandom import dataallocationrandom


# def data_per_client_per_class_random(dataallocation, pc_class):

#     random_list = [0,1,2,3,4,5,6,7,8,9]
    
#     data_per_client_per_class = []                         
#     data_temp = [0,0,0,0,0,0,0,0,0,0]

#     data_all = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]                 # 数据集中每个label的数据量最大值

#     # 初始化环境中每个client中的每个label对应的数据量，都为0
#     for m  in range(10):
#         data_per_client_per_class.append(data_temp)                            

#     # 对于每个节点分配数据量
#     for i in range(len(dataallocation)):
#         rand_class = []
#         while len(rand_class) < pc_class:
#             rand_temp = random.choice(random_list)              # 随机产生一个值
#             if data_all[rand_temp] == 0:
#                 pass
#             else:
#                 rand_class.append(rand_temp)
#         print(rand_class)
#         all_temp = 0
#         # 对于每个随机产生的class的序号，分配数据量
#         for j in range(len(rand_class)):
            
#             # 如果标签是最后一个标签时            
#             if j == (len(rand_class) - 1):    
                
#                 for a in range(10):
#                     all_temp = all_temp + data_per_client_per_class[i][a]                       # 节点中所有已经分配的标签的数据量之和                                   
#                 rand_data = dataallocation[i] - all_temp
#                 data_per_client_per_class[i][rand_class[j]] = rand_data                   # 节点i对应的标签j的数据量
#                 data_all[rand_class[j]] = data_all[rand_class[j]] - rand_data                    # 更新剩余的可分配的数据量
                
#             else:
#                 rand_data = np.random.randint(1, data_all[rand_class[j]])                       # label = j时随机产生数据量
#                 print(rand_data)
#                 data_per_client_per_class[i][rand_class[j]] = rand_data                             # 节点i对应的标签j的数据量
#                 data_all[rand_class[j]] = data_all[rand_class[j]] - rand_data                    # 更新剩余的可分配的数据量
                

#     print(data_per_client_per_class)
#     return data_per_client_per_class



# if __name__ == '__main__':
#     dataallocation = config.get("DataAllocation")
#     pc_class = 2
#     data_per_client_per_class_temp  = data_per_client_per_class_random(dataallocation, pc_class)


def data_per_client_per_class_random(dataallocation):
    data_per_client_per_class = []
    data_temp = [0,0,0,0,0,0,0,0,0,0]   
    data_all = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]                 # 数据集中每个label的数据量最大值 
    
    for a in range(len(dataallocation)):
        data_per_client_per_class.append(data_temp)
        
    for i in range(len(dataallocation)):
        c = np.random.randint(len(data_all))                    # 随机产生一个标签编号
        while dataallocation[i] > 0 :
            print(data_all[c])
            print(dataallocation[i])
            take = np.random.randint(0, min(data_all[c], dataallocation[i]))
            data_per_client_per_class[i][c] = take
            # 更新
            dataallocation[i] = dataallocation[i] - take
            data_all[c] = data_all[c] - take
            c = (c+1) % len(data_all)
    print(data_per_client_per_class)
    return data_per_client_per_class


if __name__ == '__main__':
    dataallocation = config.get("DataAllocation")
    for i in range(len(dataallocation)):
        dataallocation[i] = int(dataallocation[i] * 50000)
    data_per_client_per_class = data_per_client_per_class_random(dataallocation)



        



