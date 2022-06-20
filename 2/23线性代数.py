import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
# 线性代数的概念视频很干货，可以反复看

# 创建一个矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
"""
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]])
"""
# 对矩阵转置
print(A.T)
'''
tensor([[ 0,  4,  8, 12, 16],
        [ 1,  5,  9, 13, 17],
        [ 2,  6, 10, 14, 18],
        [ 3,  7, 11, 15, 19]])
'''


# 如果直接B=A，就只是把索引给到A。所以要B = A.clone()。
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print(A)
print(A + B)
'''
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.]])
tensor([[ 0.,  2.,  4.,  6.],
        [ 8., 10., 12., 14.],
        [16., 18., 20., 22.],
        [24., 26., 28., 30.],
        [32., 34., 36., 38.]])
'''

# 对于张量按元素乘
# 这个叫做哈达玛积，数学符号⊙
# 等价于torch.mul(a,b)
print(A*B)
'''
tensor([[  0.,   1.,   4.,   9.],
        [ 16.,  25.,  36.,  49.],
        [ 64.,  81., 100., 121.],
        [144., 169., 196., 225.],
        [256., 289., 324., 361.]])
'''


# 我们还可以指定张量沿哪⼀个轴来通过求和降低维度。
A = torch.arange(40, dtype=torch.float32).reshape(2 , 5, 4)
print(A)
'''
tensor([[[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]],

        [[20., 21., 22., 23.],
         [24., 25., 26., 27.],
         [28., 29., 30., 31.],
         [32., 33., 34., 35.],
         [36., 37., 38., 39.]]])
'''
A_sum_axis0 = A.sum(axis=0) # sum最外层维度，即2那个维度，就是把那2个加起来
print(A.sum())
print(A_sum_axis0)
print(A_sum_axis0.shape)
'''
tensor(780.)
tensor([[20., 22., 24., 26.],
        [28., 30., 32., 34.],
        [36., 38., 40., 42.],
        [44., 46., 48., 50.],
        [52., 54., 56., 58.]])
torch.Size([5, 4])
'''

# 指定axis=1将通过汇总所有列的元素降维（轴1）。因此，输入轴1的维数在输出形状中消失。
A_sum_axis1 = A.sum(axis=1) # sum 次外层维度，即5那个维度，就是把那5个加起来
print(A_sum_axis1)
print(A_sum_axis1.shape)
'''
tensor([[ 40.,  45.,  50.,  55.],
        [140., 145., 150., 155.]])
torch.Size([2, 4])
'''

# 均值mean也是一个道理

# 非降维求和
# 有时在调用函数来计算总和或均值时保持轴数不变会很有用。
sum_A = A.sum(axis=1, keepdims=True) # 也就是在那个多余加括号的地方，此时不去掉括号
print(sum_A)
'''
tensor([[[ 40.,  45.,  50.,  55.]],

        [[140., 145., 150., 155.]]])
'''

# 由于sum_A在对每行进行求和后仍保持原来的维度（只是每一维度大小不一样），我们可以通过广播将A除以sum_A。
print(A / sum_A)
'''
tensor([[[0.0000, 0.0222, 0.0400, 0.0545],
         [0.1000, 0.1111, 0.1200, 0.1273],
         [0.2000, 0.2000, 0.2000, 0.2000],
         [0.3000, 0.2889, 0.2800, 0.2727],
         [0.4000, 0.3778, 0.3600, 0.3455]],

        [[0.1429, 0.1448, 0.1467, 0.1484],
         [0.1714, 0.1724, 0.1733, 0.1742],
         [0.2000, 0.2000, 0.2000, 0.2000],
         [0.2286, 0.2276, 0.2267, 0.2258],
         [0.2571, 0.2552, 0.2533, 0.2516]]])
'''

# 如果我们想沿某个轴计算A元素的累积总和， 比如axis=0（按行计算），我们可以调用cumsum函数。 此函数不会沿任何轴降低输入张量的维度。
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A.cumsum(axis=0))
'''
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  6.,  8., 10.],
        [12., 15., 18., 21.],
        [24., 28., 32., 36.],
        [40., 45., 50., 55.]])
'''
# 即按行累加，0123后是4+0、5+1、6+2.以此类推

# 点积
x= torch.tensor([0., 1., 2., 3.])
y = torch.ones(4, dtype = torch.float32)
print(x)
print(y)
print(torch.dot(x, y))
'''
tensor([0., 1., 2., 3.])
tensor([1., 1., 1., 1.])
tensor(6.)
'''
# 向量的点积，每个对应相乘然后相加

# 矩阵向量积，使用mv（metric vector multification）
# 就是mXn的矩阵和nX1的向量会出来，mX1的向量
print(A.shape)
print(x.shape)
print(torch.mv(A, x))
'''
torch.Size([5, 4])
torch.Size([4])
tensor([ 14.,  38.,  62.,  86., 110.])
'''

# 矩阵矩阵积,使用mm（metric metric multification）
B = torch.ones(4, 3)
print(torch.mm(A, B))
'''
tensor([[ 6.,  6.,  6.],
        [22., 22., 22.],
        [38., 38., 38.],
        [54., 54., 54.],
        [70., 70., 70.]])
'''

# 范数，向量一般是用l2范数
# 各数平方开根号
u = torch.tensor([3.0, -4.0])

print(torch.norm(u))

# 也有l1范数，对绝对值求和并相加
print(torch.abs(u).sum())


# 矩阵常用的就是F范数,通过norm函数
print(torch.norm(torch.ones((4, 9))))







