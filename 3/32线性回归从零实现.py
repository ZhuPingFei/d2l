import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
# 线性回归的解可以用一个公式简单地表达出来， 这类解叫作解析解（analytical solution）
# 也就是直接让偏导为0获得结果


# ||b||2，即2范数，所有元素平方和。的偏导为   2bT
# 所以  ||y-Xw||2，的求导为先把大的求导。再把Xw求导。
# 此时X矩阵相对于w向量是常数，则用   AX求导为A   的方法
# 所以  2||y-Xw||T X

# 小批量随机梯度下降是深度学习默认的求解算法
# 两个重要的参数是批量大小和学习率

import random   # 用来做随机
import torch
from d2l import torch as d2l

'''
首先构造一个数据集
w=[2,−3.4]⊤ 、b=4.2
y=Xw+b+ϵ.
'''
# 你可以将 ϵ 视为模型预测和标签时的潜在观测误差。
# 在这里我们认为标准假设成立，即 ϵ 服从均值为0的正态分布。 为了简化问题，我们将标准差设为0.01
def synthetic_data(w, b, num_examples):  #输入参数 w b 生成样本个数
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 该函数返回从    单独的正态分布     中提取的随机数的张量，该正态分布的均值是mean，标准差是std。
    # 首先生成X，均值为0标准差为1，为标准分布。后面为返回的张量形状。二维：有n个样本，一维：跟w一样的长度。
    # 见 https://blog.csdn.net/qq_41898018/article/details/119244914
    # 标准差越大，正态越塌越广
    y = torch.matmul(X, w) + b
    # torch.matmul是tensor的乘法，输入可以是高维的。
    # 当输入是都是二维时，就是普通的矩阵乘法，和tensor.mm函数用法相同。
    # 当输入有多维时，把多出的一维作为batch提出来，其他部分做矩阵乘法。
    # 见 https://blog.csdn.net/qsmx666/article/details/105783610
    # https://blog.csdn.net/didi_ya/article/details/121158666
    y += torch.normal(0, 0.01, y.shape)
    # 加个正态分布的张量的噪声，0均值0.01标准差，形状跟y一样
    return X, y.reshape((-1, 1))
    # 返回x和  把y作为列向量返回
    # y.reshape((-1, 1)) 转换成1列，-1则表示其他维度为固定值(写出来了)，这一维度自动计算
    # 见https://blog.csdn.net/qq_29831163/article/details/90112000

true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 构建真实 w 和真实 b

features, labels = synthetic_data(true_w, true_b, 1000)
# 注意，features中的每一行都包含一个二维数据样本， labels中的每一行都包含一维标签值（一个标量）。
print('features:', features[0],'\nlabel:', labels[0])
'''
features: tensor([-1.6092, -0.5917])
label: tensor([2.9942])
'''
# 通过生成第二个特征features[:, 1]  (即遍历所有的feature，是所有的一个二元组，取所有二元组里的第二个，即拿出所有行第二列元素)和labels的散点图， 可以直观观察到两者之间的线性关系。
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# 在pytorch一些版本需要把其从计算图中detach出来才能到numpy中去

# plt.show()

# 训练模型时要对数据集进行遍历，每次抽取一小批量样本，并使用它们来更新我们的模型。
# 由于这个过程是训练机器学习算法的基础，所以有必要定义一个函数， 该函数能打乱数据集中的样本并以小批量方式获取数据。
# 在下面的代码中，我们定义一个data_iter函数
# 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量。
# 每个小批量包含一组特征和标签。
'''
读取数据集
'''
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # 获得样本大小
    indices = list(range(num_examples))
    # 生成序列，即用range输入大小生成，在用list包起来变成list

    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    # 把indices打乱，这样for的时候for这个indices就是下标随机了
    for i in range(0, num_examples, batch_size):
        # 从0到n，每次跳batch_size个大小，即进行几次(因为每次里面处理了batch_size个，所以下一次下标从batchsize个后开始)


        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        # 此时indices中是乱序列，所以按顺序取从i开始batchsize个就是一个打乱序列
        # 因为可能不是整除，所以做一个min来保证当i+batchsize下标越界时只取到最后一个

        yield features[batch_indices], labels[batch_indices]
        # 每一次通过这个产生  随机顺序 的特征  和   随机顺序的标号
        # 见 https://blog.csdn.net/mieleizhi0522/article/details/82142856

# 我们直观感受一下小批量运算：读取第一个小批量数据样本并打印。
# 每个批量的特征维度显示批量大小和输入特征数。
# 同样的，批量的标签形状与batch_size相等。
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
'''
tensor([[ 0.7614,  0.8402],
        [ 0.9188,  0.8180],
        [-0.2403, -0.5700],
        [ 0.0188, -1.1180],
        [-0.3355,  0.7889],
        [-0.3329, -2.7114],
        [-1.1159, -1.0267],
        [-1.7124,  1.3431],
        [ 0.4513, -1.4313],
        [ 0.9867,  1.0740]]) 
 tensor([[ 2.8708],
        [ 3.2666],
        [ 5.6624],
        [ 8.0183],
        [ 0.8433],
        [12.7781],
        [ 5.4409],
        [-3.7920],
        [ 9.9644],
        [ 2.5148]])
'''

'''
初始化模型参数
'''
# 设置参数初值
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


'''
定义模型
'''
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b
# b是标量，我们用一个向量加一个标量时，标量会被加到向量的每个分量上。

'''
定义损失函数
'''
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
# 此处返回的仍是向量

'''
定义优化算法
'''
def sgd(params, lr, batch_size): # 参数、学习率、batch_size
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            # 参数的梯度会存在参数的grad中，前面损失函数没有求均值，此处求均值
            param.grad.zero_()
# 在每一步中，使用从数据集中随机抽取的一个小批量，然后根据参数计算损失的梯度。
# 接下来，朝着减少损失的方向更新我们的参数。
# 下面的函数实现小批量随机梯度下降更新。
# 该函数接受模型参数集合、学习速率和批量大小作为输入。
# 每一步更新的大小由学习速率lr决定。
# 因为我们计算的损失是一个批量样本的总和，所以我们用批量大小（batch_size） 来规范化步长，这样步长大小就不会取决于我们对批量大小的选择。


'''
训练
'''
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):  # 每次十个，但是不会重复，yield会从之前断点往后执行
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，并以此计算关于[w,b]的梯度，存放到w.grad和b.grad中
        l.sum().backward() # 因为上一轮进行完SGD后，SGD函数里面调用的w和b的grad清零了梯度，所以不用在backward之前清零
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        # w b是全局变量，所以在函数中-=可以直接改变
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

'''
epoch 1, loss 0.030913
epoch 2, loss 0.000126
epoch 3, loss 0.000054
w的估计误差: tensor([0.0001, 0.0002], grad_fn=<SubBackward0>)
b的估计误差: tensor([-0.0004], grad_fn=<RsubBackward1>)
'''




































