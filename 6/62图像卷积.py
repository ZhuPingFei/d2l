import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

# 通过在图像边界周围填充零来保证有足够的空间移动卷积核，从而保持输出大小不变。
# 接下来，我们在corr2d函数中实现如上过程，该函数接受输入张量X和卷积核张量K，并返回输出张量Y。
'''
1、互相关运算
'''
import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    '''
    获得卷积核的形状，用输入的形状和卷积核形状计算输出的形状
    '''
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    '''
    两个for循环计算出一整个输出
    注：这里i到i+h，放到Y的i上
    原理上看上去是要放在中点，因为原理上是正负delta
    但是实际上这是对应好的，相当于把所有的Y向左上角平移，所以Y的输出也不会有改变
    
    Y的大小ij也是由X的大小（shape[0]和shape[1]）和K的hw计算出来的，所以不会溢出，直接for循环就好
    
    *  是对应元素相乘，所以要sum求和
    '''
    return Y


# 验证上述二维互相关运算的输出
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

print(corr2d(X, K))
'''
tensor([[19., 25.],
        [37., 43.]])
'''


'''
2、卷积层
'''
# 卷积层对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出。
# 所以，卷积层中的两个被训练的参数是卷积核权重和标量偏置。
# 就像我们之前随机初始化全连接层一样，在训练基于卷积层的模型时，我们也随机初始化卷积核权重。
#
# 基于上面定义的corr2d函数实现二维卷积层。
# 在__init__构造函数中，将weight和bias声明为两个模型参数。
# 前向传播函数调用corr2d函数并添加偏置。
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        '''
        注意：这里重写的init函数
        因为我们在初始化这个层的时候要给出 kernel_size
        卷积核的大小
        如：Conv2D((3,3))
        '''
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# 高度和宽度分别为  h  和  w  的卷积核可以被称为  h X w  卷积或  h X w  卷积核。
# 我们也将带有  h X w  卷积核的卷积层称为  h X w  卷积层。


'''
3、图像中目标的边缘检测
'''
# 如下是卷积层的一个简单应用：通过找到像素变化的位置，来检测图像中不同颜色的边缘。
# 首先，我们构造一个6X8像素的黑白图像。中间四列为黑色（0），其余像素为白色（1）。
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)
'''
tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.]])
'''
# 接下来，我们构造一个高度为1、宽度为2的卷积核K。当进行互相关运算时，如果水平相邻的两元素相同，则输出为零，否则输出为非零。
K = torch.tensor([[1.0, -1.0]])

# 现在，我们对参数X（输入）和K（卷积核）执行互相关运算。
# 如下所示，输出Y中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘，其他情况的输出为0。
Y = corr2d(X, K)
print(Y)
'''
tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])
'''
# 现在我们将输入的二维图像转置，再进行如上的互相关运算。
# 其输出如下，之前检测到的垂直边缘消失了。
# 不出所料，这个卷积核K只可以检测垂直边缘，无法检测水平边缘。

print(corr2d(X.t(), K))
'''
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
'''

'''
4、学习卷积核
'''
# 如果我们只需寻找黑白边缘，那么以上[1, -1]的边缘检测器足以。
# 然而，当有了更复杂数值的卷积核，或者连续的卷积层时，我们不可能手动设计滤波器。
# 那么我们是否可以学习由X生成Y的卷积核呢？
#
# 现在让我们看看是否可以通过仅查看“输入-输出”对来学习由X生成Y的卷积核。
# 我们先构造一个卷积层，并将其卷积核初始化为随机张量。
# 接下来，在每次迭代中，我们比较Y与卷积层输出的平方误差，然后计算梯度来更新卷积核。
# 为了简单起见，我们在此使用内置的二维卷积层，并忽略偏置。
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)
'''
这里调用pythorch的定义，参数为(输入通道数，输出通道数，卷积核，是否有偏执单元)
黑白图片通道为1，彩色图片通道为3
'''
# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
'''
这里的两个1
一个指批量大小
一个指通道数
先是批量后是通道

这里是抽象的概念
'''
lr = 3e-2  # 学习率




for i in range(10):
    '''
    共进行10轮
    '''
    Y_hat = conv2d(X)
    '''
    计算预测
    '''
    l = (Y_hat - Y) ** 2
    '''
    计算误差，平方
    '''
    conv2d.zero_grad()
    '''
    把上一轮梯度清零
    '''
    l.sum().backward()
    '''
    把误差的平方加起来求梯度
    注：这里不是一次进行了一个batchsize个，所以不用除以batchsize
    
    误差的平方和就是损失本身。
    这个损失的backward也就是损失对于各个分量求偏导。这里是对的。
    
    之前全连接层的计算时，输入输出是向量，所以一次进行10个(一个batchsize)变成矩阵和矩阵的计算
    所以也要除以batchsize
    '''
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    '''
    这里访问了weight的data，也就是权重的值
    然后对其修改，也就是用梯度下降的方法
    '''
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

'''
epoch 2, loss 1.618
epoch 4, loss 0.298
epoch 6, loss 0.061
epoch 8, loss 0.015
epoch 10, loss 0.004
'''

# 在10次迭代之后，误差已经降到足够低。现在我们来看看我们所学的卷积核的权重张量。
print(conv2d.weight.data.reshape((1, 2)))
'''
tensor([[ 0.9879, -0.9993]])
'''

































