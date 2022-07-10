import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
'''
视频缺失章节
3、1线性回归
3、4 softmax回归(这个要到书上细细的看懂里面的概念)(看第一版书)
'''
# MSEloss返回的是差的平方，即   (x-y)的平方
# 然后取平均

'''
定义一个net时
先定义形状
net = nn.Sequential(nn.Linear(2, 1))

后定义初始的参数
net[0].weight.data.normal_(0, 0.01) # 给weight写一个正态分布
net[0].bias.data.fill_(0) # 用fill来补充0


后面做优化器就是输入这个模型的参数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)




前向传播用的就是net(X)
这个net在构建时相当于    只跟输入量    打交道

所以net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
中nn.Flatten()是     对输入量    的展平成二维
'''
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat.shape)
print(len(y_hat.shape))
print(y_hat.shape[1])
