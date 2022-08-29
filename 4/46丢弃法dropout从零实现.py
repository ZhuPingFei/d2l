import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
一、dropout
'''
import torch
from torch import nn
from d2l import torch as d2l

#  该函数以dropout的概率丢弃张量输入X中的元素，
#  重新缩放剩余部分：将剩余部分除以1.0-dropout。
'''
设dropout的概率为p
则p概率x的对应位置出0
1-p概率x的对应位置不出0

此时，我们希望x的   整体的期望  不变
所以在1-p概率下x对应位置变为      h/1-p
h为该位置原值
'''

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    '''
    assert
    Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常
    如果dropout不在0和1之间，则直接报错
    见 https://www.runoob.com/python3/python3-assert.html
    '''

    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)


    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X


    mask = (torch.rand(X.shape) > dropout).float()
    '''
    rand，生成一个跟x同形状的0到1之间的 均匀随机分布
    如果大于dropout，就选出来等于1，不然就等于0.
    由此实现了dropout概率的  位置上的  丢弃。
    
    
    mask是一个0或1组成的向量，与X按元素乘
    则0的部分为0,
    1的部分，X保留原值，然后用除以(1-dropout)放大
    '''
    '''
    此处也有种写法
    X[mask] = 0
    但是对于GPU，做乘法比选择效率更高
    '''
    return mask * X / (1.0 - dropout)




# 测试
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
'''
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
tensor([[ 0.,  0.,  0.,  0.,  8., 10., 12.,  0.],
        [16.,  0., 20., 22., 24.,  0., 28., 30.]])
tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
'''

'''
二、定义模型参数
'''
# 定义具有两个隐藏层的多层感知机，每个隐藏层包含256个单元
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

'''
三、定义模型
'''
# 我们可以将暂退法应用于每个隐藏层的输出（在激活函数之后）， 并且可以为每一层分别设置暂退概率：

# 常见的技巧是在靠近输入层的地方设置    较低   的暂退概率。

# 下面的模型将第一个和第二个隐藏层的暂退概率分别设置为0.2和0.5，
# 并且暂退法只在训练期间有效。
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    '''
    # 这是一个    继承于   nn.module的Net
    # 所以我们是重写了其 forward函数
    '''


    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    '''
    # 所以在train的时候用到Net(X)时，我们调用了其forward函数.
    # 所以这个forward不是随便写的
    '''
    '''
    所以在predict时，要改变这个is_training
    '''
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)



'''
四、训练和测试
'''
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)





























































