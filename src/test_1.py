# # import numpy as np
# # a = [[[1,2],[3,4]]]
# # array_1 = np.array(a)
# # print(a)
# # b = [[[5,6],[7,8]]]
# # array_2 = np.array(b)
# # print(array_2)
# # list_1 = a + b 
# # array_3 = np.array(list_1)
# # print(array_3)
# # print("reshape",array_3.reshape)
# # print(b[0])
# # # a.append(b[0])
# # a.append(b[0])
# # print(a)

# a = [[1,3],[3,2],4,5]
# b = a
# del b[2]
# print(b)
# print(a)

# a = [[1,2],3,4,5]
# for i in range(len(a)):
#     if i == 1:
#         del a[i]
# print(a)

# a = [[1,2],3,4,5]
# for i in a:
#     if i == [1,2]:
#         del i[1]
# print(a)

# a = 0 
# b = a 
# b = 2
# del a
# print(b)

# class A(object):
#     def __init__(self, x, y=10) -> None:
#         self.x = x
#         self.y = y
    
#     def aa(self,x):
#         print(self.y)
#         print(x)
#     def aaa(self,x):
#         self.aa(x)
#         x = x + 1 
#         print(x)
#     def main(self):
#         self.aa(self.x)
#         self.aaa(self.x)

# class B(A):
#     def __init__(self,x,m) -> None:
#         # super(B,self).__init__(x) 
#         A.__init__(self,x) 
#         self.m = m  
#     def aaa(self,x):
#         self.aa(self.x)
#         x = x + 2 
#         print(x+2)
#         print("m",m)
#     def main(self):
#         super().main()

# if __name__ == "__main__":
#     x = 10
#     m = 100
#     b = B(x,m)
#     b.main()


# list_temp = []              # 初始化为空list
# client_matrix_list = [] 
# for i in range(5):
#     client_matrix_list.append([])
# print(client_matrix_list)
# print(client_matrix_list[0])
# client_matrix_list[0].append(1)
# print(client_matrix_list[0])
# print(client_matrix_list)
# list_1 = [1,2,3,4]
# for i in list_1:
#     i =  i  + 1
#     print(list_1)

# from numpy.core import test


# test_m = 10
# class A(object):
#     def __init__(self,m=test_m) -> None:
#         self.m = m
#         print(self.m)

# #     def print_m(self,m):
# #         print(m)
# # if __name__ == "__main__":
# #     am = 100
# #     a = A(m=am)
# #     print(a.m)
# #     a.print_m(a.m)        



# # list_1 = []
# # list_2 = []
# # list_2.append(list_1)
# # print("list_2.1",list_2)
# # list_1.append(2)
# # print("list_2.2",list_2)

# class A():
#     def __init__(self) -> None:
#         self.a = 1

#     def __del__(self):
#         print("类A被收回")


# A = A()
# A.b = 2
# print(A.b)
# # del A

# a = [10,23,2]
# for i in a:
#     if i == 10:
#         del i
# print(a)



import numpy as np

print(np.loadtxt("before_op.txt", delimiter=',',))