import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

'''
应该是因为这台电脑没有2块即以上的GPU，所以GPU不会有样例里面很复杂的表达形式
'''


'''
1、计算设备
'''
# 在PyTorch中，CPU和GPU可以用torch.device('cpu') 和torch.device('cuda')表示。
# 应该注意的是，cpu设备意味着所有物理CPU和内存， 这意味着PyTorch的计算将尝试使用所有CPU核心。
# 然而，gpu设备只代表一个卡和相应的显存。
# 如果有多个GPU，我们使用torch.device(f'cuda:{i}') 来表示第i块GPU（i从0开始）。
# 另外，cuda:0和cuda是等价的。
import torch
from torch import nn

print(torch.device('cpu'))
print(torch.device('cuda'))
print(torch.device('cuda:0'))
# print(torch.device('cuda:0'))没有第二块GPU
'''
cpu 
cuda
cuda:0
'''

# 我们可以查询可用gpu的数量。
print(torch.cuda.device_count())
'''
1
'''

# 现在我们定义了两个方便的函数， 这两个函数允许我们在不存在所需所有GPU的情况下运行代码。
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu())
print(try_gpu(10))# 没有10个，所以返回cpu
print(try_all_gpus())
'''
cuda:0
cpu
[device(type='cuda', index=0)]
'''


'''
2、张量与GPU
'''
# 我们可以查询张量所在的设备。 默认情况下，张量是在CPU上创建的。
x = torch.tensor([1, 2, 3])
print(x.device)
'''
cpu
'''
# 需要注意的是，无论何时我们要对多个项进行操作， 它们都必须在同一个设备上。
# 例如，如果我们对两个张量求和， 我们需要确保两个张量都位于同一个设备上，
# 否则框架将不知道在哪里存储结果，甚至不知道在哪里执行计算。

# 有几种方法可以在GPU上存储张量。
# 例如，我们可以在创建张量时指定存储设备。
# 接下来，我们在第一个gpu上创建张量变量X。 在GPU上创建的张量只消耗这个GPU的显存。
# 我们可以使用nvidia-smi命令查看显存使用情况。 一般来说，我们需要确保不创建超过GPU显存限制的数据。
X = torch.ones(2, 3, device=try_gpu())
print(X)
'''
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
'''

# 假设你至少有两个GPU，下面的代码将在第二个GPU上创建一个随机张量。
# Y = torch.rand(2, 3, device=try_gpu(1))
# Y
# 结果
# tensor([[0.1206, 0.2283, 0.4548],
#         [0.9806, 0.9616, 0.0501]], device='cuda:1')


# 复制
# Z = X.cuda(1)
# print(X)
# print(Z)
# 结果
# tensor([[1., 1., 1.],
#         [1., 1., 1.]], device='cuda:0')
# tensor([[1., 1., 1.],
#         [1., 1., 1.]], device='cuda:1')



# Y + Z
# 结果
# tensor([[1.1206, 1.2283, 1.4548],
#         [1.9806, 1.9616, 1.0501]], device='cuda:1')


# 假设变量Z已经存在于第二个GPU上。 如果我们还是调用Z.cuda(1)会发生什么？ 它将返回Z，而不会复制并分配新内存。
# Z.cuda(1) is Z
# 结果
# True
print(X.cuda(0) is X)
'''
True
'''

'''
3、神经网络与GPU
'''
# 类似地，神经网络模型可以指定设备。 下面的代码将模型参数放在GPU上。
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(X))
'''
tensor([[-0.8429],
        [-0.8429]], device='cuda:0', grad_fn=<AddmmBackward0>)
'''
# 让我们确认模型参数存储在同一个GPU上。
print(net[0].weight.data.device)
'''
cuda:0
'''





