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

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


'''
一、初始化模型参数
'''
# 单隐藏层的多层感知机
# 784个输入特征 和10个类的简单分类数据集。256个隐藏单元

# 通常，我们选择2的若干次幂作为层的宽度

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
# 784*256
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# 256
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
# 256*10
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
# 10
params = [W1, b1, W2, b2]


'''
二、激活函数
'''
# 使用relu激活函数，小于0的出0
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

'''
三、模型
'''
# 因为我们忽略了空间结构， 所以我们使用reshape将每个二维图像转换为一个长度为num_inputs的向量。
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    # batch_size * 784     @     784 * 256
    # 见https://blog.csdn.net/qq_21997625/article/details/85001493?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-2
    return (H@W2 + b2)

'''
四、损失函数
'''
loss = nn.CrossEntropyLoss(reduction='none')
# 交叉熵 同前一章


'''
五、训练
'''
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.predict_ch3(net, test_iter)






































