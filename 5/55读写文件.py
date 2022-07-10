import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
1、加载和保存张量
'''
# 对于单个张量，我们可以直接调用load和save函数分别读写它们。
# 这两个函数都要求我们提供一个名称，save要求将要保存的变量作为输入。
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
# 我们现在可以将存储在文件中的数据读回内存。
x2 = torch.load('x-file')
print(x2)
'''
tensor([0, 1, 2, 3])
'''

# 我们可以存储一个张量列表，然后把它们读回内存。
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
print((x2, y2))
'''
(tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))
'''

# 我们甚至可以写入或读取从字符串映射到张量的字典。
# 当我们要读取或写入模型中的所有权重时，这很方便。
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)
'''
{'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}
'''

'''
2、加载和保存模型参数
'''
# 保存单个权重向量（或其他张量）确实有用
# 但是如果我们想保存整个模型，并在以后加载它们， 单独保存每个向量则会变得很麻烦。
# 毕竟，我们可能有数百个参数散布在各处。
# 因此，深度学习框架提供了内置函数来保存和加载整个网络。
# 需要注意的一个重要细节是，这将保存       模型的参数         而不是保存整个模型。
# 例如，如果我们有一个3层多层感知机，我们需要          单独指定架构。
# 因为模型本身可以包含任意代码，所以模型本身难以序列化。
# 因此，为了恢复模型，我们需要用代码生成架构，然后从磁盘加载参数。
# 让我们从熟悉的多层感知机开始尝试一下。
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# 接下来，我们将模型的参数存储在一个叫做“mlp.params”的文件中。
torch.save(net.state_dict(), 'mlp.params')
# 为了恢复模型，我们实例化了原始多层感知机模型的一个备份。
# 这里我们不需要随机初始化模型参数，而是直接读取文件中存储的参数。
clone = MLP()
'''
先要生成一个新的网络，然后把磁盘上的参数传入来写掉原来初始化的参数
'''
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())
'''
eval训练模式，这个模式下在前向传播时不会有dropout(网络层中间随机扔几个值)batch normalization(把输出的值给改变一下使得控制范围)
而train模式下就会
所以train模式是训练用，而预测时候，模型不应该dp和bn，所以要置为eval模式
'''
'''
MLP(
  (hidden): Linear(in_features=20, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
)
'''

# 由于两个实例具有相同的模型参数，在输入相同的X时， 两个实例的计算结果应该相同。 让我们来验证一下。
Y_clone = clone(X)
print(Y_clone == Y)
'''
tensor([[True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True]])
'''

























































