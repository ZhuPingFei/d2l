import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
# 通常当我们处理图像时，我们希望逐渐降低隐藏表示的空间分辨率、聚集信息
# 这样随着我们在神经网络中层叠的上升，每个神经元对其敏感的感受野（输入）就越大。

# 池化层的目的
# 1、降低卷积层对位置的敏感性
# 2、同时降低对空间降采样表示的敏感性。
'''
1、最大池化层和平均池化层
'''
# 在下面的代码中的pool2d函数，我们实现汇聚层的前向传播。
# 然而，这里我们没有卷积核，输出为输入中每个区域的最大值或平均值。


from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    '''
    X: 输入(这里设其为  单样本单通道 )
    pool_size:池化层窗口大小
    mode:池化的模式
    '''
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    '''
    计算Y的大小并初始化，然后迭代赋值
    '''
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


# 我们可以构建输入张量X，验证二维最大池化层的输出。
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
'''
tensor([[4., 5.],
        [7., 8.]])
'''
# 验证平均池化层。
print(pool2d(X, (2, 2), 'avg'))
'''
tensor([[4., 5.],
        [7., 8.]])
'''


'''
2、填充和步幅
'''
# 与卷积层一样，池化层也可以改变输出形状。
# 和以前一样，我们可以通过填充和步幅以获得所需的输出形状。
# 下面，我们用深度学习框架中内置的二维最大池化层，来演示池化层中填充和步幅的使用。
# 我们首先构造了一个输入张量X，它有四个维度，其中样本数和通道数都是1。
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)
'''
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]]]])
'''

# 默认情况下，深度学习框架中的步幅与池化窗口的大小相同。
'''
即如果2X2的池化窗口，步幅为2。窗口移动没有重叠
'''
# 因此，如果我们使用形状为(3, 3)的汇聚窗口，那么默认情况下，我们得到的步幅形状为(3, 3)。
'''
这是pytorch的池化层写法
'''
pool2d = nn.MaxPool2d(3)

print(pool2d(X))
'''
tensor([[[[10.]]]])
'''
'''
4X4的矩阵用3X3的池化，就只能来一次，没有下一个
'''




# 填充和步幅可以手动设定。
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
'''
tensor([[[[ 5.,  7.],
          [13., 15.]]]])
'''
# 当然，我们可以设定一个任意大小的矩形汇聚窗口，并分别设定填充和步幅的高度和宽度。
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pool2d(X))
'''
tensor([[[[ 5.,  7.],
          [13., 15.]]]])
'''


'''
3、多个通道
'''
# 在处理多通道输入数据时，汇聚层在每个输入通道上单独运算，而不是像卷积层一样在通道上对输入进行汇总。
# 这意味着汇聚层的输出通道数与输入通道数相同。
# 下面，我们将在通道维度上连结张量X和X + 1，以构建具有2个通道的输入。
X = torch.cat((X, X + 1), 1)
print(X)
'''
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]],

         [[ 1.,  2.,  3.,  4.],
          [ 5.,  6.,  7.,  8.],
          [ 9., 10., 11., 12.],
          [13., 14., 15., 16.]]]])
'''

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
'''
tensor([[[[ 5.,  7.],
          [13., 15.]],

         [[ 6.,  8.],
          [14., 16.]]]])
'''

































































