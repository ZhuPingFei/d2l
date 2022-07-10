import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
1、不带参数的层
'''
# 首先，我们构造一个没有任何参数的自定义层。
# 下面的CenteredLayer类要从其输入中减去均值。
# 要构建它，我们只需继承基础层类并实现前向传播功能。
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        python3默认不写会自己调用父类的init
        '''

    def forward(self, X):
        '''
        所有值减去均值，即使得均值为0
        '''
        return X - X.mean()

# 让我们向该层提供一些数据，验证它是否能按预期工作。
'''
自定义层和自定义网络没有本质区别，都是nn.Module的子类
'''
layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
'''
tensor([-2., -1.,  0.,  1.,  2.])
'''

# 现在，我们可以将层作为组件合并到更复杂的模型中。
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
# 作为额外的健全性检查，我们可以在向该网络发送随机数据后，检查均值是否为0。
# 由于我们处理的是浮点数，因为存储精度的原因，我们仍然可能会看到一个非常小的非零数。
Y = net(torch.rand(4, 8))
print(Y.mean())
'''
tensor(9.3132e-10, grad_fn=<MeanBackward0>)
'''



'''
2、带参数的层
'''
# 以上我们知道了如何定义简单的层，下面我们继续定义具有参数的层，这些参数可以通过训练进行调整。
# 我们可以使用内置函数来创建参数，这些函数提供一些基本的管理功能。
# 比如管理访问、初始化、共享、保存和加载模型参数。
# 这样做的好处之一是：我们不需要为每个自定义层编写自定义的序列化程序。
#
# 现在，让我们实现自定义版本的全连接层。
# 回想一下，该层需要两个参数，一个用于表示权重，另一个用于表示偏置项。
# 在此实现中，我们使用修正线性单元作为激活函数。
# 该层需要输入参数：in_units和units，分别表示输入数和输出数。
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        '''
        重写init函数，变成接收有两个参数的函数
        '''
        super().__init__()
        '''
        参数是nn.Parameter的实例，所以我们的参数调用该类定义
        这里就用正态分布的tensor作为输入来构成nn.Parameter类
        '''
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        '''
        用该nn.Parameter类的data来计算(这个类包括data参数和grad梯度)
        '''
        return F.relu(linear)
'''
虽然在里面又nn.Parameter包起来又.data解开，看起来多此一举
其实这样就能直接访问权重(参数)，这里这个访问出来就是一个parameter的类
'''
# 接下来，我们实例化MyLinear类并访问其模型参数。
linear = MyLinear(5, 3)
print(linear.weight)
'''
Parameter containing:
tensor([[ 1.9054, -3.4102, -0.9792],
        [ 1.5522,  0.8707,  0.6481],
        [ 1.0974,  0.2568,  0.4034],
        [ 0.1416, -1.1389,  0.5875],
        [-0.7209,  0.4432,  0.1222]], requires_grad=True)
'''

# 我们可以使用自定义层直接执行前向传播计算。
print(linear(torch.rand(2, 5)))
'''
tensor([[2.4784, 0.0000, 0.8991],
        [3.6132, 0.0000, 1.1160]])
'''
'''
这里的linear是一个实例化的自定义带参数的层，5X3，在赋值时候已经初始化了，这里只是起个昵称所以用法一样
'''
# 我们还可以使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层。
'''
自定义的层继承于nn.module
所以也放的进去
'''
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))
'''
tensor([[0.],
        [0.]])
'''














































