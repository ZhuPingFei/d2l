import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
# 这里只记录几个点
# 多个张量连结在一起
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
'''
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [ 2.,  1.,  4.,  3.],
         [ 1.,  2.,  3.,  4.],
         [ 4.,  3.,  2.,  1.]]),
 tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))
'''
# 放入要连结的张量到cat中，然后dim指定连结的维度。
# 0就是1维，打开最外层括号，把元素合起来。然后再加回括号。行合并
# 1就是2维，打开两层括号，互相合并然后再加回括号。列合并

# 比较容易出错的地方。
# 广播机制。
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
'''
(tensor([[0],
         [1],
         [2]]),
 tensor([[0, 1]]))
'''
print(a+b)
'''
tensor([[0, 1],
        [1, 2],
        [2, 3]])
'''
# 当两个不同形状的张量相加时，会触发广播机制。
# 我们将两个矩阵⼴播为⼀个更⼤的3 × 2矩阵
# 矩阵a将复制列，矩阵b将复制⾏，然后再按元素相加。
# 即a变成
# tensor([[0, 0],
#         [1, 1],
#         [2, 2]])
# b变成
# tensor([[0, 1],
#         [0, 1],
#         [0, 1]])

x = torch.arange(12).reshape(3,4)
print(x[-1])
print(x[1:3])
'''
tensor([ 8,  9, 10, 11])
'''
# x[-1]拿出最后一行。即最外层维度的一个小项
# x[1:3]拿出第2和3行。3开区间取不到，取下标1、2，即第2和3行。即最外层维度的一个小项


# tensor和numpy中互换
A = x.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))
'''
<class 'numpy.ndarray'>
<class 'torch.Tensor'>
'''
# 要将大小为1的张量转换为Python标量，我们可以调用item函数或Python的内置函数
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
'''
(tensor([3.5000]), 3.5, 3.5, 3)
'''
# tensor          numpy的浮点数  python浮点数  python整数






# 下面不用

# 首先，我们介绍 n 维数组，也称为张量（tensor）
# 张量表示由一个数值组成的数组，这个数组可能有多个维度。
# 具有一个轴的张量对应数学上的向量（vector）；
# 具有两个轴的张量对应数学上的矩阵（matrix）；
# 具有两个轴以上的张量没有特殊的数学名称。
x = torch.arange(12)
print(x)
'''
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
'''


# 可以通过张量的shape属性来访问张量（沿每个轴的长度）的形状 。
print(x.shape)
'''
torch.Size([12])
'''
# 张量中元素的总数
x.numel()
'''
12
'''
# 剩下的见网站



