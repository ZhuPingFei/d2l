import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
1、多输入通道
'''
# 为了加深理解，我们实现一下多输入通道互相关运算。
# 简而言之，我们所做的就是对每个通道执行互相关操作，然后将结果相加。

from d2l import torch as d2l
'''
这个函数实现了多通道互相关操作并将结果相加
'''
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    '''
    输入的X，K都是三维张量
    把X,K    zip  起来
    for的时候就会把最外层进行遍历

    注：zip会把可迭代对象作为参数，将对象中对应元素打包成一个个元组，然后返回由这些元素组成的列表
    即把最外层[]拆了，然后把对应的x和k打包成元组，好几个元组组成列表。for列表取元组，然后分别给到这轮的x，k
    这里就是第一轮x为[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]，k为[[0.0, 1.0], [2.0, 3.0]]
    见https://mofanpy.com/tutorials/python-basic/interactive-python/lazy-usage/#Zip%E8%AE%A9%E4%BD%A0%E5%90%8C%E6%97%B6%E8%BF%AD%E4%BB%A3
    '''
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

# 我们可以构造与 图中的值相对应的输入张量X和核张量K，以验证互相关运算的输出。
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))
'''
tensor([[ 56.,  72.],
        [104., 120.]])
'''


'''
2、多输出通道
'''
# 如下所示，我们实现一个计算多个通道的输出的互相关函数。

'''
此时K为4维，k为3维
K被for的时候拆开最外的[],每个都是k(三维卷积核)
然后调用函数计算。
'''
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
'''
这些每一轮计算出来的结果——二维矩阵Y
用[]来把每一轮的计算结果保存乘一个列表

用torch.stack把他们concat起来，变成一个三维的输出，即多通道输出

torch.stack：沿着一个新维度(此处为0维开始连接，即直接前后concat)，对张量序列(接收一个张量列表作为输入)进行连接
序列中所有张量应为相同维度
'''

# 通过将核张量K与K+1（K中每个元素加1）和K+2连接起来，构造了一个具有3个输出通道的卷积核。
'''
把原有的1层K用stack构造了一个具有3个输出通道的卷积核
'''
K = torch.stack((K, K + 1, K + 2), 0)

print(K.shape)
'''
torch.Size([3, 2, 2, 2])
'''
'''
输出通道3，输入通道2，核大小2X2

因为本身co也是在最外层
'''


# 下面，我们对输入张量X与卷积核张量K执行互相关运算。
# 现在的输出包含3个通道，第一个通道的结果与先前输入张量X和多输入单输出通道的结果一致。
print(corr2d_multi_in_out(X, K))
'''
tensor([[[ 56.,  72.],
         [104., 120.]],

        [[ 76., 100.],
         [148., 172.]],

        [[ 96., 128.],
         [192., 224.]]])
'''

'''
这里就是计算多通道输出
'''


'''
3、1X1卷积层
'''
# 下面，我们使用全连接层实现1×1卷积。 请注意，我们需要对输入和输出的数据形状进行调整。
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    '''
    这里把 X    reshape成一个矩阵
    '''
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    '''
    这里是因为转置了，X时ci开头，K是ci结尾，所以是K，X
    '''
    return Y.reshape((c_o, h, w))
'''
最后reshape回去
'''

# 当执行1×1卷积运算时，上述函数相当于先前实现的互相关函数corr2d_multi_in_out。让我们用一些样本数据来验证这一点。
X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6

print(float(torch.abs(Y1 - Y2).sum()) < 1e-6)
'''
True
表示这个1X1跟原来的多输入输出通道的卷积，在1X1核上计算是一样的
'''

'''
用pytorch的封好的写法，就是nn.Conv2d中前两个维度指定输出输入
不过封装好的函数中是先输入后输出
'''


































































