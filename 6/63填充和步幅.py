import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
1、填充
'''
# 卷积神经网络中卷积核的高度和宽度通常为奇数，例如1、3、5或7。
# 选择奇数的好处是，保持空间维度的同时，我们可以在顶部和底部填充相同数量的行，在左侧和右侧填充相同数量的列。
'''
因为保持原大小，padding的行列就是核的大小减一
'''
# 对于任何二维张量X，当满足：
# 1. 卷积核的大小是奇数；
# 2. 所有边的填充行数和列数相同；
# 3. 输出与输入具有相同高度和宽度
# 则可以得出：输出Y[i, j]是通过以输入X[i, j]为中心，与卷积核进行互相关计算得到的。


# 比如，在下面的例子中，我们创建一个高度和宽度为3的二维卷积层，并在所有侧边填充1个像素。
# 给定高度和宽度为8的输入，则输出的高度和宽度也是8。
import torch
from torch import nn


# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    '''
    在不考虑通道和批量大小的时候，我们默认输入的是个矩阵
    所以我们要将其reshape，在前面加两个1
    '''
    X = X.reshape((1, 1) + X.shape)
    '''
    这里是传入我们用的函数，然后计算Y
    '''
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    '''
    出来的Y是包含前面两个11的
    这里输出省略前两个维度变成矩阵
    '''
    return Y.reshape(Y.shape[2:])

# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
'''
当kernel_size只填入一个int时候
则表示一个一个正方形的以这个int为宽高的核

padding填充同理，padding这样就是上下左右各

先高后宽
'''
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)
'''
torch.Size([8, 8])
'''
# 当卷积核的高度和宽度不同时，我们可以填充不同的高度和宽度，使输出和输入具有相同的高度和宽度。
# 在如下示例中，我们使用高度为5，宽度为3的卷积核，高度和宽度两边的填充分别为2和1。
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)
'''
torch.Size([8, 8])
'''

'''
2、步幅
'''
# 下面，我们将高度和宽度的步幅设置为2，从而将输入的高度和宽度减半。
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)
'''
torch.Size([4, 4])
'''
# 接下来，看一个稍微复杂的例子。
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)
'''
torch.Size([2, 2])
'''
'''
此处padding 与 核 不相抵消
所以结果
2=(8-3+0+3)/3
2=(8-5+1+4)/4
'''
