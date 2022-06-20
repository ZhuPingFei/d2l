import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import torch
from torch import nn
from d2l import torch as d2l
'''
一、模型
'''
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))
# Flatten展平，三维变二维，每个数据28*28，展平成784
# 其是默认为从   第一维   开始展平(不是第0维)
# 即第0维展平就是全部乘起来，变成[X]
# 第一维   展平   保留最前面的，后面乘起来  [batch_size,X]
# 因为读入的时候保存就会保存成这个样子，即batchsize在最前面
# 见 https://blog.csdn.net/Super_user_and_woner/article/details/120782656
# 见 https://www.csdn.net/tags/OtDaEgysMTkyMDAtYmxvZwO0O0OO0O0O.html
# 线性层，然后激活函数，然后线性层输出
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

'''
二、实现
'''



batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)












































































