import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
# 我们首先看一下具有单隐藏层的多层感知机。
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))
'''
tensor([[0.0147],
        [0.0776]], grad_fn=<AddmmBackward0>)
'''
'''
1、参数访问
'''
# 我们从已有模型中访问参数。
# 当通过Sequential类定义模型时，我们可以通过索引来访问模型的任意层。
# 这就像模型是一个列表一样，每层的参数都在其属性中。
# 如下所示，我们可以检查第二个全连接层的参数。
'''
net[2]拿到的就是顺序的第三个层
.state_dict()就是返回该层的参数
返回的类型是一个  有序字典   ，分别是参数的名字的参数值的键值对
'''
print(net[2].state_dict())
'''
OrderedDict([('weight', tensor([[ 0.2130, -0.3521,  0.2074,  0.3113, -0.2176, -0.1606, -0.2008,  0.3309]])), ('bias', tensor([-0.3015]))])
'''
# 输出的结果告诉我们一些重要的事情：
# 首先，这个全连接层包含两个参数，分别是该层的权重和偏置。
# 两者都存储为单精度浮点数（float32）。

# 注意，参数名称允许唯一标识每个参数，即使在包含数百个层的网络中也是如此。

'''
1、1 目标参数
'''
# 注意，每个  参数  都表示为   参数类  的一个实例。
# 要对参数执行任何操作，首先我们需要访问  底层  的数值。
# 有几种方法可以做到这一点。有些比较简单，而另一些则比较通用。
# 下面的代码从第二个全连接层（即第三个神经网络层）提取偏置，提取后返回的是一个参数类实例，并进一步访问该参数的值。

print(type(net[2].bias))# 参数类型，其类型是一个torch.nn.parameter.Parameter
print(net[2].bias)# 取参数bias
print(net[2].bias.data)# 取参数bias的值
'''
<class 'torch.nn.parameter.Parameter'>

Parameter containing:
tensor([0.0704], requires_grad=True)

tensor([0.0704])
'''
'''
.data访问参数值
.grad访问该参数的梯度
'''
# 参数是复合的对象，包含值、梯度和额外信息。
# 这就是我们需要显式参数值的原因。
# 除了值之外，我们还可以访问每个参数的梯度。
# 在上面这个网络中，由于我们还没有调用反向传播，所以参数的梯度处于初始状态。
print(net[2].weight.grad == None)
'''
True
'''

'''
1、2 一次性访问所有参数
'''
# 当我们需要对所有参数执行操作时，逐个访问它们可能会很麻烦。
# 当我们处理更复杂的块（例如，嵌套块）时，情况可能会变得特别复杂，因为我们需要递归整个树来提取每个子块的参数。
# 下面，我们将通过演示来比较访问第一个全连接层的参数和访问所有层。
print(*[(name, param.shape) for name, param in net[0].named_parameters()])

print(*[(name, param.shape) for name, param in net.named_parameters()])
'''
('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
'''
'''
net[0].named_parameters()拿出网络第一层的参数
net.named_parameters()拿出网络的所有参数
用for in读出给到一个个元组
用*解包元组，将元组分解成一个个单元
'''
'''
出来有'0.weight''2.weight'之类的，前面是层数后面是参数名，参数名唯一对应一个参数
'''
# 这为我们提供了另一种访问网络参数的方式，如下所示。
print(net.state_dict()['2.bias'].data)
'''
tensor([-0.2930])
'''
'''
可以通过   层数.参数名  访问网络参数
'''

'''
1、3 从嵌套块收集参数
'''
# 让我们看看，如果我们将多个块相互嵌套，参数命名约定是如何工作的。
# 我们首先定义一个生成块的函数（可以说是“块工厂”），然后将这些块组合到更大的块中。
def block1():
    '''
    制作一个block网络块
    '''

    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
        '''
        每次加一个block1，指定名字
        调用顺序网络nn.Sequential的add_module函数加入网络
        跟之前的1、嵌套网络层调用2、放进一个有序数组迭代加进去3、套nn.Sequential加入
        区别是可以给加入的模组命名。
        '''
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
'''
tensor([[0.4196],
        [0.4195]], grad_fn=<AddmmBackward0>)
'''
# 设计了网络后，我们看看它是如何工作的。
print(rgnet)
'''
Sequential(
  (0): Sequential(
    (block 0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)
'''
'''
相当于nn.Sequential里本身是012这样排序，然后再0层中放的也是一个nn.Sequential，里面是block1、block2这类名字
'''
# 因为层是分层嵌套的，所以我们也可以像通过嵌套列表索引一样访问它们。
# 下面，我们访问第一个主要的块中、第二个子块的第一层的偏置项。
print(rgnet[0][1][0].bias.data)
'''
tensor([-0.2726,  0.2247, -0.3964,  0.3576, -0.2231,  0.1649, -0.1170, -0.3014])
'''

'''
2参数初始化
'''
# 默认情况下，PyTorch会根据一个范围均匀地初始化权重和偏置矩阵.
# 这个范围是根据输入和输出维度计算出的。
# PyTorch的nn.init模块提供了多种预置初始化方法。
'''
2、1 内置初始化
'''
# 让我们首先调用内置的初始化器。
# 下面的代码将所有权重参数初始化为标准差为0.01的高斯随机变量，且将偏置参数设置为0
'''
m是一个module的形参。
传入module，在函数内对module进行操作
'''
def init_normal(m):
    if type(m) == nn.Linear:
        '''
        PyTorch的nn.init模块提供了多种预置初始化方法。
        .normal_，接收  所要初始化的参数、均值、方差
        .zeros_，接收 所要初始化的参数
        
        XXXX_
        下划线在python中的意思是会对读到的输入进行改变
        '''
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
'''
net来apply这个函数，即对所有   层（module）  进行该函数
假设是嵌套，那么就会嵌套遍历
即会递归调用，直到所有 层 都进行该函数 
'''
print(net[0].weight.data[0], net[0].bias.data[0])
'''
(tensor([-0.0017,  0.0232, -0.0026,  0.0026]), tensor(0.))
'''
# 我们还可以将所有参数初始化为给定的常数，比如初始化为1。
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])
'''
(tensor([1., 1., 1., 1.]), tensor(0.))
'''
# 我们还可以对某些块应用不同的初始化方法。
# 例如，下面我们使用Xavier初始化方法初始化第一个神经网络层，然后将第三个神经网络层初始化为常量值42。
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
'''
apply可以只对一层进行
'''
print(net[0].weight.data[0])
'''
这里可以看到有两层括号[]
所以data[0]是取括号里面
'''
print(net[2].weight.data)
'''
tensor([-0.4645,  0.0062, -0.5186,  0.3513])
tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
'''
'''
Xavier见  48.2 .md
即折中
输入维度乘参数方差（正向）=1 
输出维度乘参数方差（反向）=1
保证数值稳定性做的参数初始化
'''



'''
2、2 自定义初始化
'''
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5
'''
即可以在初始化里面加上奇怪的东西
这里就是报一下我们要初始化哪一层然后均匀初始化
然后如果绝对值大于等于5，则返回为true，相乘保留
否则置0
'''
net.apply(my_init)
'''
Init weight torch.Size([8, 4])
Init weight torch.Size([1, 8])
'''
print(net[0].weight[:2])
'''
:2是切片，即选取高维度的前两行
'''
print(net[0].weight)
'''
tensor([[ 8.8025,  6.4078,  0.0000, -8.4598],
        [-0.0000,  9.0582,  8.8258,  7.4997]], grad_fn=<SliceBackward0>)
'''

# 注意，我们始终可以直接设置参数。
'''
直接访问net的参数然后修改
'''
net[0].weight.data[:] += 1
'''
[:]所有值加一 
'''
net[0].weight.data[0, 0] = 42

print(net[0].weight.data[0])
'''
tensor([42.,  1.,  1.,  1.])
'''

'''
3、参数绑定
'''
# 有时我们希望在多个层间共享参数：我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。

# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
# 重要
'''
给该分享层（module）起个名字

注：这里是已经初始化为shared后，（在shared = nn.Linear(8, 8)时参数已经生成固定，该模组已创建）
然后再函数中被访问。所以是共享参数。
而一个Sequential中写的
nn.linear(同形状)
这种是单独初始化，不会共享


同之前的51的
在sequential中套一个灵活加入的层不一样
那个是class，相当于每次重新初始化
而这个是把一个确切的层的实例加入


相当于share权重
'''
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
'''
修改后，两个层也是一样
'''
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
'''
tensor([True, True, True, True, True, True, True, True])
tensor([True, True, True, True, True, True, True, True])
'''
# 这个例子表明第三个和第五个神经网络层的参数是绑定的。
# 它们不仅值相等，而且由相同的张量表示。
# 因此，如果我们改变其中一个参数，另一个参数也会改变。
# 你可能会思考：当参数绑定时，梯度会发生什么情况？
# 答案是由于模型参数包含梯度，
# 因此在反向传播期间第二个隐藏层 （即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。






































































