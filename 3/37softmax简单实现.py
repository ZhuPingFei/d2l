import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
# 通过pytorch的nn模组实现
import torch
from torch import nn
from d2l import torch as d2l
batch_size = 256
if __name__ == "__main__":
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 拿到数据迭代器

'''
一、初始化模型参数
'''
# softmax回归的输出层是一个全连接层。
# 因此，为了实现我们的模型， 我们只需在Sequential中添加一个带有10个输出的全连接层。
# 同样，在这里Sequential并不是必要的， 但它是实现深度模型的基础。 我们仍然以均值0和标准差0.01随机初始化权重。

# PyTorch不会   隐式地调整输入的形状   。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
'''nn.Flatten()第零维度保留，其他维度全部展开成为向量'''

'''
# 对应从零实现中的reshape(98行)


# Flatten，任何维度的tensor变成2D的tensor。
# 因为在神经网络中，输入为一批数据，第一维为batch，通常要把一个数据拉成一维，而不是将一批数据拉为一维。
# 所以torch.nn.Flatten()默认从第二维开始平坦化。
# 见 https://blog.csdn.net/Super_user_and_woner/article/details/120782656
'''




'''
定义一个初始化参数的函数，m指的是当前的layer。
每调用一次就是对当前层进行一次初始化
'''
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        '''
        如果这个层是nn.linear全连接层
        那么就把他的weight初始化成   均值为0方差为0.01的正态分布
        
        nn是跟net没有关系，是一个库
        nn.init是独立于我们构建的net的一个方法
        
        接收三个参数，tensor、mean=0、std=1
        这里tensor是m.weight,就是把迭代m(该层网络)的w初始化。mean没写就是默认0
        见 https://zhuanlan.zhihu.com/p/101313762
        '''
        '''
        本节使用的方法与线性回归有不同，见33第112行
        '''
net.apply(init_weights)
'''
apply是模型(多个网络的组合)的一个内置函数。
apply函数接收一个函数，(这个函数是一个接收网络层的函数)，将函数递归的运用在每个子模块上，用来对网络层进行操作，即递归的运用到每个层上
见 https://blog.csdn.net/qq_37025073/article/details/106739513
'''



'''
二、重新审视Softmax的实现
'''
'''
二、定义损失函数
'''
# 在从零实现中，我们计算了模型的输出，然后将此输出送入交叉熵损失。
# 从数学上讲，这是一件完全合理的事情。 然而，从计算角度来看，指数可能会造成数值稳定性问题。

# 有时候如果 Ok(原始输出)中的一些数值非常大， 那么 exp(ok)可能大于数据类型容许的最大数字，即上溢（overflow）。
# 这将使分母或分子变为inf（无穷大）， 最后得到的是0、inf或nan（不是数字）

# 。。。见网页书

loss = nn.CrossEntropyLoss(reduction='none')
'''
这个函数直接实现了softmax后在计算交叉熵(y作为一个01向量来对y_hat取值)，即正确归类的预测概率

然后把这个-log而产出loss以便于优化(因为这个是预测概率不会大于1，所以-log一定非负)
'''
'''
真正计算过程见https://blog.csdn.net/qq_44523137/article/details/120557043
其实nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合版本。
'''
# 见https://zhuanlan.zhihu.com/p/431283706
# reduction，默认mean
# 它的可选参数有三个，分别是'none','mean'和'sum'。其中，假设在一个batch内进行计算的话，
#
# 'none'代表的是batch内的每个元素都会计算一个损失，返回的结果还是一个batch；
# 'mean’代表的是是否进行平均，一个batch只返回一个；
# 'sum’代表的是将batch内的loss相加，一个batch也是只返回一个；
# 其中，参数weight代表的是是否加入权重，如果加入的话，它的size大小需要和目标类的个数是一样的。
'''
三、优化算法
'''
# 在这里，我们使用学习率为0.1的小批量随机梯度下降作为优化算法。 这与我们在线性回归例子中的相同，这说明了优化器的普适性。
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
'''
同33的136行
'''

'''
四、训练
'''
# 接下来我们调用 3.6节中 定义的训练函数来训练模型。
'''
3.6中写的train是对于自写模型和已有模型都适配的，可以直接用
'''
if __name__ == "__main__":
    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
















































