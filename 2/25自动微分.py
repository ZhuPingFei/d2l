import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
<x,w>的内积，对于这个的导数，对w的偏导就是xT，对x的偏导就是wT

||b||2，即2范数，所有元素平方和。的偏导为   2bT
'''

x = torch.arange(4.0)
print(x)
'''
tensor([0., 1., 2., 3.])
'''
# 在我们计算 y关于 x 的梯度之前，我们需要一个地方来存储梯度。
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
print(x.grad)  # 默认值是None
'''
None
'''
# 此时没有设立y，也没有进行y的backward所以没有梯度

#  注意，一个标量函数关于向量 x的梯度是向量，并且与 x 具有相同的形状。

# 现在让我们计算 y。 y是x的点积，是一个标量。
y = 2 * torch.dot(x, x)
print(y)
'''
tensor(28., grad_fn=<MulBackward0>)
'''
# 因为这是一个隐式的构造的计算部，所以他有一个求梯度的函数grad_fn = <MulBackward0>来告诉你 y 是从 x 计算过来的

#  接下来，我们通过调用反向传播函数来自动计算y关于x每个分量的梯度，并打印这些梯度。
y.backward()
print(x.grad)
'''
tensor([ 0.,  4.,  8., 12.])
'''
# 函数 y=2xT x 关于 x的梯度应为 4x
# 让我们快速验证这个梯度是否计算正确。
print(x.grad == 4 * x)
'''
tensor([True, True, True, True])
'''

# 现在让我们计算x的另一个函数。
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
# 为什么pytorch会默认累计梯度
# 通常需要累积梯度的时刻，假设有一个批量，反传过程中很耗内存，对于一个大的批量可能存在算不下的情况
# 1、可以把比如128的批量分成四个64累积。
# 2、一个weight在不同模型间share时有好处
# 3、多个loss反向传播的时候需要累加梯度

x.grad.zero_() # 即把0写入x的梯度，即把x的梯度清零   因为x存在上一个y的梯度，即使换了y梯度还在
y = x.sum()  # 相当于是x1+x2这种，所以梯度是一个1向量
y.backward()
print(x.grad)
'''
tensor([1., 1., 1., 1.])
'''

# 当我们的y并不是一个标量
# 比如
x.grad.zero_()
y = x * x
# 理论上此时y的back_fn是一个矩阵
# 但是在机器学习中很少去对一个向量的函数求导
# 大部分情况只是对标量进行求导
# 所以我们对y进行求和在来进行求导

# 解释：loss通常是标量，如果loss是向量很麻烦，向量关于矩阵的loss是一个矩阵。
# 矩阵在往下走就是四维矩阵，神经网络一深张量就特别大算不出来


y.sum().backward()
print(x.grad)

# 将某些计算移动到记录的计算图之外
# 这对于之后做神经网络时候要把一些参数固定时很有帮助
x.grad.zero_()
y = x * x
u = y.detach() # 把y   detach掉，即把y当成一个常数而不是x的函数，做成u
# 此时u相对于系统就是一个常数（y还是函数），值为x*x
z = u * x
# z就相当于一个常数乘以x
z.sum().backward()
print(x.grad == u)
'''
tensor([True, True, True, True])
'''
# 由于记录了y的计算结果，我们可以随后在y上调用反向传播， 得到y=x*x关于的x的导数，即2*x。
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)
'''
tensor([True, True, True, True])
'''

# 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
# 因为每次在算的时候，torch会把整个计算图在背后给你存下来，那么把计算图倒着回去做一遍就可以得到他的矩阵
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)  # 都是乘以常数，所以梯度就是d/a
'''
tensor(True)
'''
# 优点，控制流做得好
# 缺点，运行慢




