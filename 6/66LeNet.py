import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
# 用深度学习框架实现此类模型非常简单。
# 我们只需要实例化一个Sequential块并将需要的层连接在一起。
import torch
from torch import nn
from d2l import torch as d2l

'''
1、LeNet
'''

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),nn.Sigmoid(),
    # '''
    # 原始数据集是32X32，我们这里的输入是28X28，所以两边各pad两行
    # '''
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    # 这里没有加padding，所以14X14变成了10X10，14-5+1
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    # 第一维批量保持住，后面全拉成一个一个维度
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

# 我们对原始模型做了一点小改动，去掉了最后一层的高斯激活。
# 除此之外，这个网络与最初的LeNet-5一致。

# 下面，我们将一个大小为28×28的单通道（黑白）图像通过LeNet。
# 通过在每一层打印输出的形状，我们可以检查模型，以确保其操作与我们期望的一致。
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)


'''
用nn.sequential做的，所以可以对里面的每一层做迭代
输出层的名字
每次用X保存层的输出，来展示输出的形状
'''
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)

'''
Conv2d output shape:         torch.Size([1, 6, 28, 28])
Sigmoid output shape:        torch.Size([1, 6, 28, 28])
AvgPool2d output shape:      torch.Size([1, 6, 14, 14])
Conv2d output shape:         torch.Size([1, 16, 10, 10])
Sigmoid output shape:        torch.Size([1, 16, 10, 10])
AvgPool2d output shape:      torch.Size([1, 16, 5, 5])
Flatten output shape:        torch.Size([1, 400])
Linear output shape:         torch.Size([1, 120])
Sigmoid output shape:        torch.Size([1, 120])
Linear output shape:         torch.Size([1, 84])
Sigmoid output shape:        torch.Size([1, 84])
Linear output shape:         torch.Size([1, 10])
'''
'''
这个过程可以认为是一个模式
不断把空间信息变小、压缩
通道数增多，把不同的抽出来的压缩的信息放在不同的通道里面
最后mlp把这些所有的模式拿出来
通过一个多层感知机训练到最后的输出
'''
# 请注意，在整个卷积块中，与上一层相比，每一层特征的高度和宽度都减小了。
# 第一个卷积层使用2个像素的填充，来补偿5×5卷积核导致的特征减少。
#
# 相反，第二个卷积层没有填充，因此高度和宽度都减少了4个像素。
# 随着层叠的上升，通道的数量从输入时的1个，增加到第一个卷积层之后的6个，再到第二个卷积层之后的16个。
# 同时，每个汇聚层的高度和宽度都减半。
'''
因为汇聚层用核的stride大小跟核大小一样，各个不重复。
则2X2的核使得高宽减半
'''
# 最后，每个全连接层减少维数，最终输出一个维数与结果分类数相匹配的输出。

'''
2、模型训练
'''
# 现在我们已经实现了LeNet，让我们看看LeNet在Fashion-MNIST数据集上的表现。
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 虽然卷积神经网络的参数较少，但与深度的多层感知机相比，它们的计算成本仍然很高
# 因为每个参数都参与更多的乘法。
# 如果你有机会使用GPU，可以用它加快训练。

# 为了进行评估，我们需要对 3.6节中描述的evaluate_accuracy函数进行轻微的修改。
# 由于完整的数据集位于内存中，因此在模型使用GPU计算数据集之前，我们需要将其复制到显存中。
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    # 两个累加器，即在初始化一个2维的metric来保存过程中产生的变量
    with torch.no_grad():
        '''
        见：https://blog.csdn.net/sazass/article/details/116668755
        '''
        for X, y in data_iter:
            '''
            如果X是列表，则所有的放进去
            如果X是张量，则直接放进去即可
            '''
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
            '''
            metric[0] / metric[1]
            前面是每次accuracy的累加
            后面是每次y的个数的累加
            除一下就是正确率
            '''
    return metric[0] / metric[1]


# 为了使用GPU，我们还需要一点小改动。
# 与 3.6节中定义的train_epoch_ch3不同，在进行正向和反向传播之前，我们需要将每一小批量数据移动到我们指定的设备（例如GPU）上。
#
# 如下所示，训练函数train_ch6也类似于 3.6节中定义的train_ch3。
# 由于我们将实现多层神经网络，因此我们将主要使用高级API。
# 以下训练函数假定从高级API创建的模型作为输入，并进行相应的优化。
#
# 我们使用在 4.8.2.2节中介绍的Xavier随机初始化模型参数。
# 与全连接层一样，我们使用交叉熵损失函数和小批量随机梯度下降。
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    '''
    如果是全连接层或者卷积层，那么就用xavier这个函数来初始化
    根据输入输出大小，使得在用随机输入时候，输出和输入方差是差不多的
    保证不要在正向反向过程梯度爆炸活梯度消失
    '''

    net.apply(init_weights)
    '''
    网络的每一层都应用这个函数初始化参数
    '''


    print('training on', device)
    '''
    看下在用什么设备
    '''

    net.to(device)
    '''
    把网络放到GPU上
    '''


    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    '''
    优化方法用SGD随机梯度下降
    接收网络的参数作为输入项，在函数中对网络参数进行更新
    '''



    loss = nn.CrossEntropyLoss()
    '''
    多分类问题，损失用交叉熵损失，确保是要的相关类的差距
    y是one-hot，与输出交叉熵，取得其预测为正确分类的概率
    将之与y相减来作为损失
    '''



    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    '''
    画动画
    '''


    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        '''
        对每次一整个数据集前向反向做迭代
        '''
        metric = d2l.Accumulator(3)
        # 三个累加器，即在初始化一个3维的metric来保存过程中产生的变量
        net.train()
        '''
        把网络设置为训练模式
        '''
        for i, (X, y) in enumerate(train_iter):
            '''
            在这轮epoch中，对数据取多次batch_size大小的数据来参与
            '''
            timer.start()
            optimizer.zero_grad()
            '''
            对梯度清零
            '''
            X, y = X.to(device), y.to(device)
            '''
            输入输出挪到GPU上
            '''
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            '''
            前向
            计算损失
            求梯度
            优化函数取梯度更新权重
            '''
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            '''
            把需要拿来计算的结果放进一个累加器metric中，后面用来计算loss、accuracy
            '''
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            '''
            metric[2]是X.shape[0]，也就是这轮X被拿的batch_size的大小
            loss就是平均loss乘以batchsize，后面计算在除以即可
            accuracy就是所有的accuracy加起来除以总数就是平均的accuracy了
            '''
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
            '''
            i时batchsize的迭代，当i满足对应的个数时(一个batch_size一打)
            把训练集loss和训练集精度加进去，之后打印用
            
            训练集一个batchsize一打(最后不够了也一打)，测试集一个epoch一打，训练集的数据密一点
            '''
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        '''
        计算每个epoch测试集精度
        '''
        animator.add(epoch + 1, (None, None, test_acc))
        '''
        把每个epoch的测试集精度放进去，之后打印用
        '''
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


# 现在，我们训练和评估LeNet-5模型。
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
'''
loss 0.451, train acc 0.833, test acc 0.812
89216.6 examples/sec on cuda:0
'''
'''
try_gpu()
函数接收一个int
函数中为
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
    
即：用torch.cuda.device_count()判断有没有对应号码gpu。
如果有几号GPU，则返回torch.device()
里面的f''是表示{}中的i为变量，不是字符i。
即返回   torch.device(cuda:0)


这里就作为找到的GPU输入，网络和参数都在这个GPU上运行
'''
'''
test_acc是在epoch里面用写的方法算出来的，每个epoch一存，代表每次epoch的测试精度

在计算的方法中，会把网络设置为   评估模式  net.eval()
'''
# 在计算的方法中，会把网络设置为   评估模式  net.eval()
'''
train_acc和train_loss是每轮batchsize会把所有的存进metric里
每个batch的数据都会与前面batch相累加，统一做计算。
(会有许多中间的并非一个epoch的部分数据上不同batchsize后的输出)(每个batchsize都会改变参数)
(每个batch累加器不清零，所以这个数据不是    实时的最新的参数对于最新的batchsize个数据上的精度)
最后每个节点用作展示的会是最后一个batch的数据，也就是一整个epoch的训练精度和训练损失

然后每个epoch开始，又会重新设置参数清零。

在网络训练中，每个epoch计算完测试精度，最上面又把网络置为    训练模式   net.train()
'''
# 在网络训练中，每个epoch计算完测试精度，最上面又把网络置为    训练模式   net.train()

'''
train 和  eval 在前向中有不同
model.train()的作用是启用 Batch Normalization 和 Dropout。
model.eval()的作用是不启用 Batch Normalization 和 Dropout。
model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。
对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。

BN层就是压缩数据的区间
'''
